"""
예측 csv -> 케이스별 지표 / 상세 결과. torch 없이도 돌아가도록 evaluate.py와 분리했다.

케이스별 평가 구성 (세 케이스 모두 같은 무무 test를 음성 샘플로 사용)
    유유 = hate                 + non_hate
    유무 = image_hate_text_safe + non_hate
    무유 = text_hate_image_safe + non_hate
    전체 = test 전체 (네 케이스 모두)
    무무 = non_hate만 -> Specificity = TN / (TN + FP)
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from common import CASE_NAME_KO, NEG_CASE, POS_CASES

NAN = float("nan")


def _div(a, b):
    return a / b if b else NAN


def binary_metrics(df):
    """df: columns y_true(0/1), y_pred(0/1), score. 혐오=양성(1)."""
    y, p, s = df["y_true"].values, df["y_pred"].values, df["score"].values
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    two = len(set(y)) == 2
    return {
        "n": int(len(df)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        # --- 논문 메인 지표 ---
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, pos_label=1, zero_division=0),
        "recall": recall_score(y, p, pos_label=1, zero_division=0),
        "f1_hate": f1_score(y, p, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "auroc": roc_auc_score(y, s) if two else NAN,
        "specificity": _div(tn, tn + fp),
        # --- 보조 지표 ---
        "f1_nonhate": f1_score(y, p, pos_label=0, zero_division=0),
        "precision_nonhate": _div(tn, tn + fn),          # = NPV
        "balanced_accuracy": balanced_accuracy_score(y, p) if two else NAN,
        "mcc": matthews_corrcoef(y, p) if two else NAN,
        "auprc": average_precision_score(y, s) if two else NAN,
        "fpr": _div(fp, fp + tn),                        # 무해를 혐오로 잘못 판단한 비율
        "fnr": _div(fn, fn + tp),                        # 혐오를 놓친 비율
        "pred_hate_rate": float(p.mean()),               # 혐오라고 예측한 비율
        # --- confusion matrix ---
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # --- 점수 분포 (score = logp혐오 - logp비혐오) ---
        "score_mean_pos": float(s[y == 1].mean()) if (y == 1).any() else NAN,
        "score_mean_neg": float(s[y == 0].mean()) if (y == 0).any() else NAN,
    }


METRIC_KEYS = list(binary_metrics(pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "score": [0.0, 1.0]})).keys())


def _neg_only_row(neg):
    tn = int((neg["y_pred"] == 0).sum()); fp = int((neg["y_pred"] == 1).sum())
    row = {k: NAN for k in METRIC_KEYS}
    row.update({"n": int(len(neg)), "n_pos": 0, "n_neg": int(len(neg)),
                "specificity": tn / len(neg), "fpr": fp / len(neg),
                "pred_hate_rate": fp / len(neg),
                "tn": tn, "fp": fp, "fn": 0, "tp": 0,
                "score_mean_neg": float(neg["score"].mean())})
    return row


def case_subsets(pred_df):
    """(case, 한글명, 부분 df) 목록. 무무는 음성만 있는 부분집합."""
    out = []
    for case in POS_CASES:
        sub = pred_df[pred_df["label"].isin([case, NEG_CASE])]
        if (sub["label"] == case).sum():
            out.append((case, CASE_NAME_KO[case], sub))
    neg = pred_df[pred_df["label"] == NEG_CASE]
    if len(neg):
        out.append((NEG_CASE, CASE_NAME_KO[NEG_CASE], neg))
    out.append(("all", "전체", pred_df))
    return out


def case_metrics(pred_df):
    rows = []
    for case, ko, sub in case_subsets(pred_df):
        m = _neg_only_row(sub) if case == NEG_CASE else binary_metrics(sub)
        rows.append({"case": case, "case_ko": ko, **m})
    return rows


def label_breakdown(pred_df):
    """네 케이스 각각에 대해 '정답을 맞힌 비율' (케이스 단독). 유유/유무/무유는 탐지율(recall),
    무무는 specificity와 같다. 어떤 조합을 특히 놓치는지 한눈에 보는 용도."""
    rows = []
    for case in POS_CASES + [NEG_CASE]:
        d = pred_df[pred_df["label"] == case]
        if d.empty:
            continue
        rows.append({
            "case": case, "case_ko": CASE_NAME_KO[case], "n": int(len(d)),
            "correct_rate": float((d["y_pred"] == d["y_true"]).mean()),
            "pred_hate_rate": float(d["y_pred"].mean()),
            "n_wrong": int((d["y_pred"] != d["y_true"]).sum()),
            "score_mean": float(d["score"].mean()), "score_median": float(d["score"].median()),
            "score_std": float(d["score"].std()) if len(d) > 1 else NAN,
            "prob_hate_mean": float(d["prob_hate"].mean()) if "prob_hate" in d else NAN,
        })
    return rows


def roc_points(pred_df):
    """케이스별 ROC 곡선 점 (나중에 ROC 그림이 필요하면 이걸로 그리면 됨)."""
    rows = []
    for case, ko, sub in case_subsets(pred_df):
        if sub["y_true"].nunique() < 2:
            continue
        fpr, tpr, thr = roc_curve(sub["y_true"], sub["score"])
        rows += [{"case": case, "case_ko": ko, "fpr": a, "tpr": b, "threshold": c} for a, b, c in zip(fpr, tpr, thr)]
    return rows


def save_all(pred_df, out_dir, extra):
    """상세 결과 파일을 전부 저장하고 case_metrics 행을 반환."""
    enc = "utf-8-sig"
    rows = case_metrics(pred_df)

    m = pd.DataFrame(rows)
    for k, v in extra.items():
        m[k] = v
    m.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding=enc)

    pd.DataFrame(label_breakdown(pred_df)).to_csv(os.path.join(out_dir, "per_label_breakdown.csv"), index=False, encoding=enc)

    cm = [{"case": r["case"], "case_ko": r["case_ko"], "TN": r["tn"], "FP": r["fp"], "FN": r["fn"], "TP": r["tp"],
           # 행 기준 정규화 (실제 혐오 중 몇 %를 혐오로 / 실제 비혐오 중 몇 %를 비혐오로)
           "TP_rate(recall)": _div(r["tp"], r["tp"] + r["fn"]), "FN_rate": _div(r["fn"], r["tp"] + r["fn"]),
           "FP_rate": _div(r["fp"], r["fp"] + r["tn"]), "TN_rate(specificity)": _div(r["tn"], r["fp"] + r["tn"])}
          for r in rows]
    pd.DataFrame(cm).to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=False, encoding=enc)

    pd.DataFrame(roc_points(pred_df)).to_csv(os.path.join(out_dir, "roc_points.csv"), index=False, encoding=enc)

    # 오분류 목록 (4.4 오분류 분석용): 확신을 갖고 틀린 순서대로
    err = pred_df[pred_df["y_true"] != pred_df["y_pred"]].copy()
    err["error_type"] = np.where(err["y_true"] == 1, "FN", "FP")
    err["case_ko"] = err["label"].map(CASE_NAME_KO)
    err["confidence_wrong"] = np.where(err["y_true"] == 1, -err["score"], err["score"])
    err.sort_values(["error_type", "case_ko", "confidence_wrong"], ascending=[True, True, False]) \
       .to_csv(os.path.join(out_dir, "errors.csv"), index=False, encoding=enc)
    return rows


def print_rows(rows):
    fmt = lambda v: "   -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    print(f"{'Case':<6}{'n':>6}  {'Acc':>6} {'P':>6} {'R':>6} {'F1h':>6} {'F1m':>6} {'AUROC':>6} {'Spec':>6}   [TN,FP,FN,TP]")
    for r in rows:
        print(f"{r['case_ko']:<6}{r['n']:>6}  {fmt(r['accuracy'])} {fmt(r['precision'])} {fmt(r['recall'])} "
              f"{fmt(r['f1_hate'])} {fmt(r['f1_macro'])} {fmt(r['auroc'])} {fmt(r['specificity'])}   "
              f"[{r['tn']},{r['fp']},{r['fn']},{r['tp']}]")
