"""
같은 게시물 정답으로 MM과 TO/IO의 정오를 짝지어 비교하는 McNemar 검정 (4.2절 교체용).

seed마다 조건(11/10/01/00) × 비교(MM-TO, MM-IO) 8개 검정을 하고 Benjamini–Hochberg로 보정한다.
임계값은 eval_common.py가 고른 값(results/.../per_run.csv)을 그대로 쓴다.

Usage
    py mcnemar_post.py --pred "preds/varco5seed_*.csv" --per-run results/varco5seed/per_run.csv --out results/varco5seed
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import eval_common


def mcnemar_exact(b, c):
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = eval_common.load(sorted({p for pat in args.pred for p in glob.glob(pat)}))
    df = df[df["split"] == "test"]
    thr = pd.read_csv(args.per_run).set_index(eval_common.RUN_KEYS)["threshold"]
    df["correct"] = [
        int((s > thr[(m, st, sd, im)]) == y)
        for s, m, st, sd, im, y in zip(df["score"], df["model"], df["setting"], df["seed"], df["input_mode"], df["y"])
    ]
    wide = df.pivot_table(index=["model", "setting", "seed", "condition", "sample_id"],
                          columns="input_mode", values="correct").reset_index()

    rows = []
    for (model, setting, seed), g in wide.groupby(["model", "setting", "seed"]):
        tests = []
        for cond in ("11", "10", "01", "00"):
            gc = g[g["condition"] == cond]
            for other in ("TO", "IO"):
                b = int(((gc["MM"] == 1) & (gc[other] == 0)).sum())  # MM만 맞힘
                c = int(((gc["MM"] == 0) & (gc[other] == 1)).sum())  # 단일 입력만 맞힘
                tests.append({"model": model, "setting": setting, "seed": seed, "condition": cond,
                              "comparison": f"MM vs {other}", "mm_only": b, "other_only": c,
                              "acc_mm": gc["MM"].mean(), f"acc_other": gc[other].mean(),
                              "p": mcnemar_exact(b, c)})
        q = bh([t["p"] for t in tests])
        for t, qq in zip(tests, q):
            t["q_bh"] = qq
            t["sig"] = qq < 0.05
            t["winner"] = ("MM" if t["mm_only"] > t["other_only"] else "single") if t["sig"] else "-"
        rows += tests
    res = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "mcnemar_post.csv"), index=False, encoding="utf-8-sig")

    summ = (res.groupby(["model", "setting", "condition", "comparison"], sort=False)
            .agg(n_seeds=("seed", "size"), sig_mm=("winner", lambda w: int((w == "MM").sum())),
                 sig_single=("winner", lambda w: int((w == "single").sum())),
                 acc_mm=("acc_mm", "mean"), acc_other=("acc_other", "mean")).reset_index())
    summ.to_csv(os.path.join(args.out, "mcnemar_post_summary.csv"), index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
