"""
자연도 분석 표 (3.7.3, 3.7.4)

validity : 판정자 타당성. 실제 게시물(조건 11·비혐오, 300건)과 무작위 결합(조건 10·01, 300건)을
           자연도 점수로 구분하는 AUROC를 seed마다 구한다. 사전 기준: 0.65 미만이면 구분하지 못한다고 본다.
stratify : 조건 10·01 각각에서 가장 어색한 50건과 가장 자연스러운 50건의 판정 결과 차이(어색 − 자연).
           지표는 Multimodal 탐지율, Image Only·Text Only 혐오 판정 비율(각 모델의 BA 임계값 적용 결과).
           seed별 차이의 평균을 보고하고, seed 안에서 표본을 복원추출하는 부트스트랩(5,000회)으로 95% CI.
compare  : 무작위 결합(R)과 자연도 반영 결합(C)의 세 구성(R→R, R→C, C→C) 비교.
           조건별 탐지율, 비혐오 FPR, Image Only 정확도(조건 11·10)의 seed 평균 ± 표준편차.
           조건 간 pairwise 대비는 main_experiment/analysis/matched_design_analysis.py로 구한다.

입력
    nat/nat_seed<N>.csv                                 (score_naturalness.py)
    <pred-root>/seed<N>/test_predictions_<label>_<mode>.csv  (evaluate.py, pred_label에 임계값 반영됨)

Usage
    python naturalness_report.py validity --nat-dir nat --out results
    python naturalness_report.py stratify --nat-dir nat --pred-root ../main_experiment/analysis/outputs_v2_bal --out results
    python naturalness_report.py compare --config "R->R=../main_experiment/analysis/outputs_v2_bal" \
        --config "R->C=outputs_r2c" --config "C->C=outputs_compat" --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REAL = ("hate", "non_hate")
CROSS = {"10": "image_hate_text_safe", "01": "text_hate_image_safe"}
MODES = {"Multimodal 혐오 탐지율": "full", "Image Only 혐오 판정 비율": "image_only",
         "Text Only 혐오 판정 비율": "text_only"}


def load_nat(nat_dir, seed):
    return pd.read_csv(os.path.join(nat_dir, f"nat_seed{seed}.csv"), dtype={"id": str})


def pred_hate(pred_root, seed, label, mode):
    df = pd.read_csv(os.path.join(pred_root, f"seed{seed}", f"test_predictions_{label}_{mode}.csv"),
                     dtype={"id": str})
    return df.set_index("id")["pred_label"].eq("hate").astype(int)


def to_markdown(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write(df, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8")
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(df) + "\n")
    print(df.to_string(index=False))


def validity(args):
    rows = []
    for seed in args.seeds:
        nat = load_nat(args.nat_dir, seed)
        real = nat[nat.label.isin(REAL)]

        def auc(cross):
            y = np.r_[np.ones(len(real)), np.zeros(len(cross))]
            return roc_auc_score(y, np.r_[real.nat_score, cross.nat_score])

        rows.append({"seed": seed,
                     "전체(조건 10+01)": auc(nat[nat.label.isin(CROSS.values())]),
                     "조건 10": auc(nat[nat.label == CROSS["10"]]),
                     "조건 01": auc(nat[nat.label == CROSS["01"]]),
                     "실제 혐오 평균": nat.loc[nat.label == "hate", "nat_score"].mean(),
                     "실제 비혐오 평균": nat.loc[nat.label == "non_hate", "nat_score"].mean()})
    df = pd.DataFrame(rows)
    mean = df.drop(columns="seed").mean().to_dict()
    df = pd.concat([df, pd.DataFrame([{"seed": "평균", **mean}])], ignore_index=True).round(3)
    write(df, args.out, "validity")
    weak = [r["seed"] for r in rows if r["전체(조건 10+01)"] < 0.65]
    if weak:
        print(f"사전 기준(AUROC 0.65) 미만 seed: {weak}")


def stratify(args):
    rng = np.random.default_rng(args.boot_seed)
    rows = []
    for cond, label in CROSS.items():
        for metric, mode in MODES.items():
            seed_diffs, boots = [], []
            for seed in args.seeds:
                nat = load_nat(args.nat_dir, seed)
                sub = nat[nat.label == label].sort_values("nat_score", kind="mergesort")
                pred = pred_hate(args.pred_root, seed, label, mode)
                awk = pred.loc[sub.id.head(args.n_group)].to_numpy()
                natural = pred.loc[sub.id.tail(args.n_group)].to_numpy()
                seed_diffs.append(awk.mean() - natural.mean())
                ia = rng.integers(0, len(awk), (args.n_boot, len(awk)))
                inat = rng.integers(0, len(natural), (args.n_boot, len(natural)))
                boots.append(awk[ia].mean(1) - natural[inat].mean(1))
            boot_mean = np.mean(boots, axis=0)
            lo, hi = np.percentile(boot_mean, [2.5, 97.5])
            rows.append({"조건": cond, "지표": metric,
                         "seed별 차이": ", ".join(f"{d:+.2f}" for d in seed_diffs),
                         "평균 차이(%p)": round(100 * np.mean(seed_diffs), 1),
                         "95% CI(%p)": f"({100 * lo:+.1f}, {100 * hi:+.1f})",
                         "CI가 0 제외": bool(lo > 0 or hi < 0),
                         "양(+)인 seed": f"{sum(d > 0 for d in seed_diffs)}/{len(seed_diffs)}"})
    write(pd.DataFrame(rows), args.out, "stratify")


def config_metrics(pred_root, seed):
    det = {c: pred_hate(pred_root, seed, lab, "full").mean()
           for c, lab in (("11", "hate"), ("10", "image_hate_text_safe"), ("01", "text_hate_image_safe"))}
    io = pd.concat([pred_hate(pred_root, seed, "hate", "image_only"),
                    pred_hate(pred_root, seed, "image_hate_text_safe", "image_only")])
    return {"조건 11 탐지율": det["11"], "조건 10 탐지율": det["10"], "조건 01 탐지율": det["01"],
            "비혐오 오탐률(FPR)": pred_hate(pred_root, seed, "non_hate", "full").mean(),
            "Image Only 정확도, 조건 11과 10": io.mean()}


def compare(args):
    configs = [c.split("=", 1) for c in args.config]
    per_seed = {name: pd.DataFrame([config_metrics(root, s) for s in args.seeds], index=args.seeds)
                for name, root in configs}
    metrics = list(next(iter(per_seed.values())).columns)
    table = pd.DataFrame({"지표": metrics})
    for name, df in per_seed.items():
        table[name] = [f"{df[m].mean():.3f} ± {df[m].std(ddof=1):.3f}" for m in metrics]
    write(table, args.out, "compare")
    long = pd.concat([df.assign(config=name, seed=df.index) for name, df in per_seed.items()])
    long.to_csv(os.path.join(args.out, "compare_per_seed.csv"), index=False, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validity", "stratify", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
        p.add_argument("--out", default="results")
        if name in ("validity", "stratify"):
            p.add_argument("--nat-dir", default="nat")
        if name == "stratify":
            p.add_argument("--pred-root", required=True)
            p.add_argument("--n-group", type=int, default=50)
            p.add_argument("--n-boot", type=int, default=5000)
            p.add_argument("--boot-seed", type=int, default=0)
        if name == "compare":
            p.add_argument("--config", action="append", required=True, help="이름=예측 폴더 (여러 번)")
    args = ap.parse_args()
    {"validity": validity, "stratify": stratify, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
