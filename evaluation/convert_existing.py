"""
이미 있는 예측 결과를 eval_common.py의 공통 형식 CSV로 바꾼다 (재학습·재추론 없음).

1) varco-seeds : 원 논문의 VARCO 5 seed (unified_pipeline/analysis/GIVEDOUTPUTS/outputs_v2_bal)
   - MM: test_predictions_all_full.csv (600건)
   - TO/IO: 조건 11·10·01 파일만 있고 비혐오(00) TO/IO 예측은 저장되지 않았다.
     그런데 test의 비혐오 텍스트는 모두 조건 10에, 비혐오 이미지는 모두 조건 01에 다시 쓰이고,
     TO(회색 이미지+텍스트)·IO(이미지+텍스트 없는 프롬프트)는 입력이 같으면 점수도 같다
     (VARCO에서 같은 입력끼리 점수 차이 0.0 확인). 그래서 00의 TO는 같은 텍스트의 조건 10 TO,
     00의 IO는 같은 이미지의 조건 01 IO 점수를 복사한다. 복사한 행은 derived_from에 원본 id를 남긴다.
     (--data-dir을 주지 않으면 복사하지 않고, TO/IO의 FPR·AUROC·Acc는 '-'로 남는다)
   - 임계값: threshold.json의 new_threshold (val BA 최대, 학습 때 고른 값)

2) model-compare : 4.3절 모델 비교 (Model_compare/model_compare_results.zip, seed 42)
   - MM: eval/, TO: eval_mask_image/, IO: eval_mask_text/  (각 600건, 비혐오 포함)
   - val: val_predictions.csv (MM만, 208건) -> eval_common이 BA 임계값을 다시 고른다

3) zeroshot : vlm_compare/zeroshot_score.py의 출력 (results_zeroshot/<model>.csv, 7개 모델)
   - test 600건 + val 208건 x MM/TO/IO. condition "non" -> "00", input_mode -> MM/TO/IO
   - 임계값은 넣지 않는다 -> eval_common이 입력 방식별 val에서 BA 최대 임계값을 고른다

Usage
    py convert_existing.py varco-seeds   --root ../unified_pipeline/analysis/GIVEDOUTPUTS/outputs_v2_bal \
        --data-dir ../unified_pipeline/analysis/data_v2 --out preds
    py convert_existing.py model-compare --root inputs/model_compare/outputs --out preds
    py convert_existing.py zeroshot      --root ../vlm_compare/results_zeroshot --out preds
"""
import argparse
import json
import os

import pandas as pd

COND = {"hate": "11", "image_hate_text_safe": "10", "text_hate_image_safe": "01", "non_hate": "00"}
# 원 논문 학습(focal loss, modality dropout, 5 seed)과 4.3절 비교 학습(plain CE, seed 42)은 레시피가 달라 구분한다
SETTING_MAIN = "fine-tuned (main)"
SETTING_CMP = "fine-tuned (comparison)"
SETTING_ZS = "zero-shot"
ZS_DISPLAY = {"varco": "VARCO-VISION-2.0-1.7B", "hyperclovax": "HyperCLOVAX-SEED-Vision-3B",
              "bllossom": "Bllossom-AICA-5B", "ax": "A.X-4.0-VL-Light", "qwen3vl": "Qwen3-VL-8B",
              "internvl3": "InternVL3-8B", "llavaov": "LLaVA-OneVision-7B"}
ZS_COND = {"11": "11", "10": "10", "01": "01", "non": "00"}
ZS_MODE = {"multimodal": "MM", "text_only": "TO", "image_only": "IO"}
COLS = ["sample_id", "condition", "input_mode", "score", "model", "setting", "seed", "split", "threshold",
        "derived_from"]


def frame(df, id_col, label_col, mode, model, setting, seed, split="test", threshold=None):
    out = pd.DataFrame({
        "sample_id": df[id_col].astype(str),
        "condition": df[label_col].map(COND),
        "input_mode": mode,
        "score": df["score"].astype(float),
    })
    if out["condition"].isna().any():
        raise ValueError(f"모르는 label: {set(df[label_col]) - set(COND)}")
    out["model"], out["setting"], out["seed"], out["split"] = model, setting, seed, split
    out["threshold"] = threshold
    out["derived_from"] = ""
    return out[COLS]


def fill_nonhate_unimodal(df, test_jsonl):
    """00의 TO/IO 점수를 같은 텍스트(조건 10 TO)·같은 이미지(조건 01 IO)에서 복사한다."""
    recs = {}
    with open(test_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["id"]] = r
    added = []
    for mode, src_cond, key in (("TO", "10", "source_text"), ("IO", "01", "image_path")):
        src = df[(df["input_mode"] == mode) & (df["condition"] == src_cond)]
        score_by_key = {}
        for _, r in src.iterrows():
            k = recs[r["sample_id"]][key]
            if k in score_by_key and abs(score_by_key[k][0] - r["score"]) > 1e-4:
                raise ValueError(f"{mode}: 같은 입력인데 점수가 다름 ({k[:30]})")
            score_by_key[k] = (r["score"], r["sample_id"])
        mm00 = df[(df["input_mode"] == "MM") & (df["condition"] == "00")]
        for _, r in mm00.iterrows():
            k = recs[r["sample_id"]][key]
            if k not in score_by_key:
                raise ValueError(f"{mode}: 비혐오 {r['sample_id']}의 입력이 조건 {src_cond}에 없음")
            row = r.copy()
            row["input_mode"], row["score"], row["derived_from"] = mode, score_by_key[k][0], score_by_key[k][1]
            added.append(row)
    return pd.concat([df, pd.DataFrame(added)], ignore_index=True)


def varco_seeds(root, out_dir, seeds, data_dir=None):
    for seed in seeds:
        d = os.path.join(root, f"seed{seed}")
        with open(os.path.join(d, "threshold.json"), encoding="utf-8") as f:
            thr = json.load(f)["new_threshold"]
        parts = []
        mm = pd.read_csv(os.path.join(d, "test_predictions_all_full.csv"))
        parts.append(frame(mm, "id", "true_label", "MM", "VARCO-VISION-2.0-1.7B", SETTING_MAIN, seed, threshold=thr))
        for case in ("hate", "image_hate_text_safe", "text_hate_image_safe"):
            for suffix, mode in (("text_only", "TO"), ("image_only", "IO")):
                p = os.path.join(d, f"test_predictions_{case}_{suffix}.csv")
                parts.append(frame(pd.read_csv(p), "id", "true_label", mode,
                                   "VARCO-VISION-2.0-1.7B", SETTING_MAIN, seed, threshold=thr))
        df = pd.concat(parts, ignore_index=True)
        if data_dir:
            df = fill_nonhate_unimodal(df, os.path.join(data_dir, f"seed{seed}", "test.jsonl"))
        path = os.path.join(out_dir, f"varco5seed_seed{seed}.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print(path, df.groupby("input_mode").size().to_dict())


def model_compare(root, out_dir):
    for key in sorted(os.listdir(root)):
        d = os.path.join(root, key)
        if not os.path.isfile(os.path.join(d, "run_info.json")):
            continue
        with open(os.path.join(d, "run_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        name, seed = info["model_display"], int(info["seed"])
        parts = [frame(pd.read_csv(os.path.join(d, "val_predictions.csv")), "id", "label", "MM",
                       name, SETTING_CMP, seed, split="val")]
        for sub, mode in (("eval", "MM"), ("eval_mask_image", "TO"), ("eval_mask_text", "IO")):
            p = os.path.join(d, sub, "predictions.csv")
            if os.path.exists(p):
                parts.append(frame(pd.read_csv(p), "id", "label", mode, name, SETTING_CMP, seed))
        df = pd.concat(parts, ignore_index=True)
        path = os.path.join(out_dir, f"modelcompare_{key}.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print(path, df.groupby(["split", "input_mode"]).size().to_dict())


def zeroshot(root, out_dir):
    for fname in sorted(os.listdir(root)):
        key, ext = os.path.splitext(fname)
        if ext != ".csv":
            continue
        df = pd.read_csv(os.path.join(root, fname), dtype={"condition": str, "sample_id": str})
        out = pd.DataFrame({
            "sample_id": df["sample_id"],
            "condition": df["condition"].map(ZS_COND),
            "input_mode": df["input_mode"].map(ZS_MODE),
            "score": df["score"].astype(float),
        })
        if out[["condition", "input_mode"]].isna().any().any():
            raise ValueError(f"{fname}: 모르는 condition/input_mode 값")
        out["model"], out["setting"], out["seed"], out["split"] = ZS_DISPLAY.get(key, key), SETTING_ZS, 0, df["split"]
        path = os.path.join(out_dir, f"zeroshot_{key}.csv")
        out.to_csv(path, index=False, encoding="utf-8")
        print(path, out.groupby(["split", "input_mode"]).size().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["varco-seeds", "model-compare", "zeroshot"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="preds")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--data-dir", default=None, help="varco-seeds: data_v2 (비혐오 TO/IO 복원용)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.source == "varco-seeds":
        varco_seeds(args.root, args.out, [int(s) for s in args.seeds.split(",")], args.data_dir)
    elif args.source == "model-compare":
        model_compare(args.root, args.out)
    else:
        zeroshot(args.root, args.out)


if __name__ == "__main__":
    main()
