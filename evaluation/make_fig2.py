"""
그림 2 교체본: 조건별·입력 방식별 혐오 판정 비율 (게시물 단위 정답, VARCO 5 seed 평균±표준편차).
원본 그림(fig1_per_case_all.png)과 같은 가로세로 비율로 저장한다.

Usage
    py make_fig2.py --summary results/varco5seed/summary.csv --out results/fig2_post_level.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = [("MM", "Multimodal", "#2a78d6", None),
          ("TO", "Text only (image grayed)", "#eb6834", "//"),
          ("IO", "Image only (no text)", "#1baf7a", "\\\\")]
CONDS = [("det_11", "Cond. 11\n(I harmful + T harmful)"),
         ("det_10", "Cond. 10\n(I harmful + T safe)"),
         ("det_01", "Cond. 01\n(I safe + T harmful)"),
         ("fpr", "Non-hate 00\n(false positive)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/varco5seed/summary.csv")
    ap.add_argument("--out", default="results/fig2_post_level.png")
    args = ap.parse_args()

    s = pd.read_csv(args.summary).set_index("input_mode")
    x = np.arange(len(CONDS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 5.94), dpi=200)  # 원본 1039x882 비율
    ink, muted = "#0b0b0b", "#52514e"
    for k, (mode, label, color, hatch) in enumerate(SERIES):
        means = [s.loc[mode, c] for c, _ in CONDS]
        stds = [s.loc[mode, f"{c}_std"] for c, _ in CONDS]
        pos = x + (k - 1) * w
        ax.bar(pos, means, w - 0.03, color=color, hatch=hatch, edgecolor="white", linewidth=0.6,
               label=label, zorder=2)
        ax.errorbar(pos, means, yerr=stds, fmt="none", ecolor=muted, elinewidth=0.9, capsize=2.5, zorder=3)
        for xi, m, sd in zip(pos, means, stds):
            ax.text(xi, m + sd + 0.015, f"{m + 1e-9:.2f}", ha="center", va="bottom", fontsize=7.5, color=ink)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CONDS], fontsize=8.5, color=ink)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Rate judged hateful", fontsize=9.5, color=ink)
    ax.set_title("Hateful-judgment rate by condition and input\n"
                 "(post-level answer: 11/10/01 = hate, 00 = non-hate; mean ± SD over 5 seeds)",
                 fontsize=9.5, color=ink)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9994")
    ax.tick_params(colors=muted, labelsize=8.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(args.out, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
