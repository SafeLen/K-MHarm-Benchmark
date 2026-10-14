"""원본 / 이미지 가림 / 텍스트 가림 결과를 모아 모달리티 기여도 표를 만든다."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CASES = {"hate": "유유", "image_hate_text_safe": "유무", "text_hate_image_safe": "무유", "non_hate": "무무"}
MODELS = ["varco", "hyperclovax", "bllossom", "ax", "qwen3vl"]
COND = {"full": "eval", "img_masked": "eval_mask_image", "txt_masked": "eval_mask_text"}

rows, samples = [], []
for m in MODELS:
    dfs = {c: pd.read_csv(p).set_index("id") for c, sub in COND.items()
           if os.path.exists(p := os.path.join("outputs", m, sub, "predictions.csv"))}
    if "full" not in dfs or len(dfs) == 1:
        continue
    base = dfs["full"][["label", "y_true", "score", "y_pred"]].rename(columns={"score": "score_full", "y_pred": "pred_full"})
    for c in ("img_masked", "txt_masked"):
        if c in dfs:
            base[f"score_{c}"] = dfs[c]["score"]
            base[f"pred_{c}"] = dfs[c]["y_pred"]
            base[f"drop_{c}"] = base["score_full"] - base[f"score_{c}"]
    base["model"] = m
    samples.append(base.reset_index())
    neg = base[base.label == "non_hate"]
    for lab, ko in CASES.items():
        d = base[base.label == lab]
        r = {"model": m, "case": ko, "n": len(d)}
        for c in ("full", "img_masked", "txt_masked"):
            col = "score_full" if c == "full" else f"score_{c}"
            if col not in base:
                continue
            pc = "pred_full" if c == "full" else f"pred_{c}"
            r[f"score_mean_{c}"] = d[col].mean()
            r[f"pred_hate_rate_{c}"] = d[pc].mean()
            if lab != "non_hate":
                y = np.r_[np.ones(len(d)), np.zeros(len(neg))]
                r[f"auroc_{c}"] = roc_auc_score(y, np.r_[d[col].values, neg[col].values])
        for c in ("img_masked", "txt_masked"):
            if f"drop_{c}" in d:
                r[f"score_drop_{c}"] = d[f"drop_{c}"].mean()
        rows.append(r)

out = pd.DataFrame(rows)
out.to_csv("outputs/modality_ablation_summary.csv", index=False, encoding="utf-8-sig")
pd.concat(samples).to_csv("outputs/modality_ablation_per_sample.csv", index=False, encoding="utf-8-sig")
cols = [c for c in ["model", "case", "pred_hate_rate_full", "pred_hate_rate_img_masked", "pred_hate_rate_txt_masked",
                    "auroc_full", "auroc_img_masked", "auroc_txt_masked",
                    "score_drop_img_masked", "score_drop_txt_masked"] if c in out]
print(out[cols].round(3).to_string(index=False))
print("\n저장: outputs/modality_ablation_summary.csv, outputs/modality_ablation_per_sample.csv")
