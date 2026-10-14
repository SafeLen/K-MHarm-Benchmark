"""
이미 학습·평가가 끝난 seed들의 판정 임계값을 다른 기준으로 validation에서 다시 고르고,
evaluate.py가 저장한 test 예측 csv의 score에 그 임계값을 고정 적용해 pred_label만 다시 쓴다.

학습/평가를 다시 돌리지 않는다: score(logp_hate - logp_nonhate)는 임계값과 무관하므로
test는 저장된 score를 그대로 쓰고, validation(seed당 ~136개)만 다시 채점한다. validation
분리는 train_lora.py의 split_train_val을 run_info.json의 val_frac/seed로 다시 호출해서
학습 때와 똑같이 재현한다(val_size가 다르면 중단).

배경: --case-pairs 데이터는 val도 정답의 약 3/4이 혐오라, 기본 f1 기준이 임계값을 혐오
쪽으로 크게 낮춘다(seed1 실측: -1.88 -> test non_hate FPR 34.7%, 같은 모델에 0.0을
적용하면 12.7%). balanced_accuracy 기준은 두 클래스를 같은 비중으로 본다.

Usage (analysis/ 에서):
    python reselect_threshold.py --dataset-root ../../image_dataset_small \
        --data-dir data_v2 --in-root outputs_v2 --out-root outputs_v2_bal \
        --seeds 1,2,3,4,5 --strategy balanced_accuracy
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration  # noqa: E402

import train_lora  # noqa: E402


def load_model(adapter_dir, device):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(adapter_dir, use_fast=True)
    base = LlavaOnevisionForConditionalGeneration.from_pretrained(
        train_lora.MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()
    return model, processor


def rewrite_predictions(in_dir, out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)
    n_files = 0
    for path in sorted(glob.glob(os.path.join(in_dir, "test_predictions_*.csv"))):
        if path.endswith(("_fn.csv", "_fp.csv")):
            continue
        df = pd.read_csv(path)
        df["pred_label"] = (df["score"] > threshold).map({True: "hate", False: "non_hate"})
        df.to_csv(os.path.join(out_dir, os.path.basename(path)), index=False)
        n_files += 1
    return n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../../image_dataset_small")
    ap.add_argument("--data-dir", default="data_v2")
    ap.add_argument("--in-root", default="outputs_v2")
    ap.add_argument("--out-root", default="outputs_v2_bal")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--strategy", default="balanced_accuracy",
                    choices=["f1", "precision_constrained", "balanced_accuracy"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = []
    for seed in seeds:
        in_dir = os.path.join(args.in_root, f"seed{seed}")
        adapter_dir = os.path.join(in_dir, "lora-adapter")
        with open(os.path.join(adapter_dir, "run_info.json"), encoding="utf-8") as f:
            run_info = json.load(f)

        train_jsonl = os.path.join(args.data_dir, f"seed{seed}", "train.jsonl")
        records = [json.loads(l) for l in open(train_jsonl, encoding="utf-8")]
        _, val_records = train_lora.split_train_val(records, run_info["val_frac"], run_info["seed"])
        if len(val_records) != run_info["val_size"]:
            raise RuntimeError(f"seed{seed}: 재현한 val 크기({len(val_records)})가 학습 때"
                               f"({run_info['val_size']})와 다릅니다 -- 중단합니다.")

        model, processor = load_model(adapter_dir, device)
        threshold, val_f1 = train_lora.select_threshold_on_val(
            model, processor, val_records, args.dataset_root, device, strategy=args.strategy,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        out_dir = os.path.join(args.out_root, f"seed{seed}")
        n_files = rewrite_predictions(in_dir, out_dir, threshold)
        info = {
            "seed": seed, "strategy": args.strategy,
            "old_strategy": run_info.get("threshold_strategy"),
            "old_threshold": run_info.get("selected_threshold"),
            "new_threshold": threshold, "val_f1_at_new_threshold": val_f1,
            "val_size": len(val_records), "n_prediction_files": n_files,
        }
        with open(os.path.join(out_dir, "threshold.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"[seed{seed}] {info}")
        summary.append(info)

    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
