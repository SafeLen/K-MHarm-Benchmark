"""
학습된 LoRA 어댑터로 test 전체를 한 번 채점하고, 케이스별 지표를 계산한다.

    python evaluate.py --model varco --adapter-dir outputs/varco

채점 방식 (모든 모델 동일)
    score = logp("혐오") - logp("비혐오")  (teacher-forced log-likelihood)
    pred  = 혐오 if score > threshold, threshold는 train_lora.py가 val에서 고른 값을 그대로 사용

출력 (--adapter-dir/eval/)
    metrics.csv / .json       케이스별 전체 지표 (Acc, P, R, F1(hate), F1(macro), AUROC, Specificity,
                              + F1(비혐오), NPV, Balanced Acc, MCC, AUPRC, FPR, FNR, 혐오 예측 비율,
                              TN/FP/FN/TP, 점수 평균) + 파라미터 수, 샘플당 추론 시간, 임계값, GPU
    confusion_matrix.csv      케이스별 TN/FP/FN/TP 개수 + 행 기준 비율
    per_label_breakdown.csv   네 케이스 각각 단독으로 맞힌 비율 / 혐오 예측 비율 / 점수 분포
    errors.csv                오분류(FN/FP) 전체 목록, 확신을 갖고 틀린 순서 (4.4 오분류 분석용)
    roc_points.csv            케이스별 ROC 곡선 점
    predictions.csv           표본별 점수/예측
    eval_summary.txt          콘솔에 출력된 요약
"""
import argparse
import json
import math
import os
import time

import pandas as pd
import torch

import image_prep
from common import is_hate
import contextlib
import io

from metrics import print_rows, save_all
from models import get_adapter, hate_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["varco", "hyperclovax", "bllossom", "ax", "qwen3vl"])
    ap.add_argument("--adapter-dir", default=None, help="기본값: outputs/<model>")
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--test-jsonl", default="data/test.jsonl")
    ap.add_argument("--threshold", type=float, default=None,
                    help="미지정 시 run_info.json의 selected_threshold(val에서 고른 값) 사용")
    ap.add_argument("--limit", type=int, default=None, help="스모크 테스트용")
    args = ap.parse_args()

    adapter_dir = args.adapter_dir or os.path.join("outputs", args.model)
    out_dir = os.path.join(adapter_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_info = {}
    ri_path = os.path.join(adapter_dir, "run_info.json")
    if os.path.exists(ri_path):
        run_info = json.load(open(ri_path, encoding="utf-8"))
    threshold = args.threshold if args.threshold is not None else run_info.get("selected_threshold")
    if threshold is None:
        threshold = 0.0
        print("[경고] run_info.json에 임계값이 없어 0.0 사용")
    print(f"임계값(val에서 선택, test에 고정): {threshold:.4f}")

    adapter = get_adapter(args.model, device)
    base = adapter.load()
    base_params = sum(p.numel() for p in base.parameters())
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()

    records = [json.loads(l) for l in open(args.test_jsonl, encoding="utf-8")]
    if args.limit:
        records = records[:args.limit]

    # 워밍업 1회 (첫 호출의 CUDA 초기화 시간이 추론 시간에 섞이지 않도록)
    for r in records[:1]:
        img = image_prep.prepare_image(os.path.join(args.dataset_root, r["image_path"]), "test", sample_id=r["id"])
        if img is not None:
            hate_score(adapter, model, img, r["prompt"])

    rows, times = [], []
    for i, r in enumerate(records):
        img = image_prep.prepare_image(os.path.join(args.dataset_root, r["image_path"]), "test", sample_id=r["id"])
        if img is None:
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        score, lp_h, lp_n = hate_score(adapter, model, img, r["prompt"])
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        rows.append({
            "id": r["id"], "label": r["label"], "target": r["target"], "y_true": is_hate(r),
            "score": score, "prob_hate": 1 / (1 + math.exp(-max(min(score, 50.0), -50.0))),
            "logp_hate": lp_h, "logp_nonhate": lp_n,
            "y_pred": int(score > threshold), "source_text": r.get("source_text", ""),
        })
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(records)} 채점")

    pred = pd.DataFrame(rows)
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")

    latency_ms = 1000 * sum(times) / len(times)
    extra = {
        "model_key": args.model, "model_display": adapter.display,
        "base_params": base_params, "lora_trainable_params": run_info.get("lora_trainable_params"),
        "latency_ms_per_sample": latency_ms, "latency_ms_median": 1000 * sorted(times)[len(times) // 2],
        "threshold": threshold, "n_scored": len(pred), "n_skipped_images": len(records) - len(pred),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    metric_rows = save_all(pred, out_dir, extra)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": metric_rows, **extra, "run_info": run_info}, f, ensure_ascii=False, indent=2, default=float)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"[{adapter.display}]  threshold={threshold:.4f}")
        print_rows(metric_rows)
        print(f"\n파라미터 {base_params / 1e9:.2f}B | 샘플당 추론 시간 {latency_ms:.1f} ms "
              f"(혐오/비혐오 두 후보 채점 포함, {extra['gpu']})")
    print(buf.getvalue())
    with open(os.path.join(out_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"저장 폴더: {out_dir}")


if __name__ == "__main__":
    main()
