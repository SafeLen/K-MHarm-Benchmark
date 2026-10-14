"""
모델별 결과 -> 세 모델 합본 csv들 (outputs/ 아래)
    results_table.csv              논문 표 형태 (모델 x 케이스, 주요 지표)
    all_metrics_full.csv           모든 지표 원본 (보조 지표 포함)
    all_per_label_breakdown.csv    케이스 단독 정답률 / 점수 분포
    all_confusion_matrix.csv       케이스별 TN/FP/FN/TP
    all_epoch_log.csv              epoch별 train/val loss
    all_run_info.csv               학습 설정 / 임계값 / 학습 시간

    python collect_results.py

평가가 끝난 모델만 모아준다. 모델 하나만 끝났어도 실행 가능.
열: 모델, Case, n, Acc, P, R, F1(hate), F1(macro), AUROC, Specificity, TN, FP, FN, TP,
    파라미터(B), LoRA 학습 파라미터, 샘플당 추론시간(ms), 임계값, GPU
무무 행은 Specificity만, 유유/유무/무유 행은 Specificity 빈칸 (세 행 모두 같은 무무를 쓰므로 무무 행에 한 번만 기록).
"""
import argparse
import json
import os

import pandas as pd

from common import CASE_NAME_KO, POS_CASES

MODEL_ORDER = ["varco", "hyperclovax", "bllossom", "ax", "qwen3vl"]
CASE_ORDER = POS_CASES + ["non_hate", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--out", default="outputs/results_table.csv")
    args = ap.parse_args()

    rows = []
    for key in MODEL_ORDER:
        path = os.path.join(args.outputs_root, key, "eval", "metrics.json")
        if not os.path.exists(path):
            continue
        m = json.load(open(path, encoding="utf-8"))
        by_case = {r["case"]: r for r in m["cases"]}
        for case in CASE_ORDER:
            if case not in by_case:
                continue
            r = by_case[case]
            rows.append({
                "모델": m["model_display"],
                "Case": "전체" if case == "all" else CASE_NAME_KO[case],
                "n": r["n"],
                "Acc": r["accuracy"], "P": r["precision"], "R": r["recall"],
                "F1(hate)": r["f1_hate"], "F1(macro)": r["f1_macro"], "AUROC": r["auroc"],
                "Specificity": r["specificity"] if case == "non_hate" else None,
                "TN": r["tn"], "FP": r["fp"], "FN": r["fn"], "TP": r["tp"],
                "파라미터(B)": round(m["base_params"] / 1e9, 2),
                "LoRA 학습 파라미터": m.get("lora_trainable_params"),
                "샘플당 추론시간(ms)": round(m["latency_ms_per_sample"], 1),
                "임계값": m.get("threshold"),
                "GPU": m.get("gpu"),
            })

    if not rows:
        print("평가 결과가 없습니다. evaluate.py를 먼저 실행하세요.")
        return
    df = pd.DataFrame(rows)
    num_cols = ["Acc", "P", "R", "F1(hate)", "F1(macro)", "AUROC", "Specificity"]
    df[num_cols] = df[num_cols].astype(float).round(3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")  # 엑셀에서 한글 안 깨지게
    print(df.drop(columns=["GPU"]).to_string(index=False))

    # 모든 지표 원본을 모델 구분 없이 한 파일로 (metrics.csv / per_label_breakdown.csv / run_info.csv 합본)
    root = os.path.dirname(args.out) or "."
    saved = [args.out]
    for fname, out_name in (("eval/metrics.csv", "all_metrics_full.csv"),
                            ("eval/per_label_breakdown.csv", "all_per_label_breakdown.csv"),
                            ("eval/confusion_matrix.csv", "all_confusion_matrix.csv"),
                            ("epoch_log.csv", "all_epoch_log.csv"),
                            ("run_info.csv", "all_run_info.csv")):
        parts = []
        for key in MODEL_ORDER:
            p = os.path.join(args.outputs_root, key, fname)
            if os.path.exists(p):
                parts.append(pd.read_csv(p).assign(model_key=key))
        if parts:
            path = os.path.join(root, out_name)
            pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
            saved.append(path)
    print("\n저장:\n  " + "\n  ".join(saved))


if __name__ == "__main__":
    main()
