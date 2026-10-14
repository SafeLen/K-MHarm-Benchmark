"""
공통 평가 스크립트 (B 담당, C와 공유).

모든 모델·설정을 같은 정의로 채점한다. 정답은 "게시물 단위"로 통일한다.
    조건 11, 10, 01 -> 혐오(1),  조건 00(비혐오 게시물) -> 비혐오(0)
입력 방식(MM / TO / IO)과 상관없이 정답은 바뀌지 않는다. 예를 들어 조건 10을 TO로
넣었을 때도 정답은 혐오이므로, 텍스트만 보는 모델은 조건 10을 놓치게 된다.

입력 CSV (여러 개 가능, 열 이름 고정)
    필수: sample_id, condition, input_mode, score 또는 pred
        condition  : 11 | 10 | 01 | 00   (hate / image_hate_text_safe /
                     text_hate_image_safe / non_hate 표기도 받는다)
        input_mode : MM | TO | IO        (multimodal / text_only / image_only 표기도 받는다)
        score      : 클수록 혐오. VLM은 log p(혐오) - log p(비혐오)
        pred       : 0/1 (점수가 없는 모델용. 이 경우 AUROC는 비워 둔다)
    선택: model, setting(zero-shot | fine-tuned | trained 등), seed,
          split(val | test, 기본 test), threshold

임계값 (run = model × setting × seed × input_mode)
    1) pred가 있으면 그대로 쓴다
    2) threshold 열이 있으면 그 값을 쓴다
    3) 같은 run의 val 행이 있으면 val에서 Balanced accuracy 최대 임계값을 고른다
    4) 같은 model·setting·seed의 MM val 행만 있으면 그 임계값을 쓴다 (val_MM)
    5) 아무것도 없으면 0 (log-odds 0 = 혐오/비혐오 확률이 같은 점)
    어떤 규칙을 썼는지는 thr_source 열에 남긴다. test로 임계값을 고르지 않는다.

출력 (--out 디렉터리)
    per_run.csv  : run별 Acc, BA, AUROC, Macro-F1, FPR(조건 00), Det@11/10/01
    summary.csv  : model·setting·input_mode별 seed 평균과 표준편차
    summary.md   : summary를 표로

Usage
    py eval_common.py --pred preds/varco_seed*.csv --out results/varco
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

CONDITION_ALIASES = {
    "11": "11", "hate": "11",
    "10": "10", "image_hate_text_safe": "10",
    "01": "01", "1": "01", "text_hate_image_safe": "01",
    "00": "00", "0": "00", "non_hate": "00",
}
MODE_ALIASES = {
    "mm": "MM", "multimodal": "MM", "full": "MM",
    "to": "TO", "text_only": "TO", "text": "TO",
    "io": "IO", "image_only": "IO", "image": "IO",
}
RUN_KEYS = ["model", "setting", "seed", "input_mode"]
METRICS = ["acc", "ba", "auroc", "macro_f1", "fpr", "det_11", "det_10", "det_01"]


def load(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"condition": str, "sample_id": str})
        df["_file"] = os.path.basename(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for col, default in (("model", "model"), ("setting", ""), ("seed", 0), ("split", "test")):
        if col not in df:
            df[col] = default
    df["seed"] = df["seed"].fillna(0).astype(int)
    df["split"] = df["split"].fillna("test").str.lower()
    df["setting"] = df["setting"].fillna("")

    df["condition"] = df["condition"].astype(str).str.strip().map(lambda c: CONDITION_ALIASES.get(c, c))
    bad = sorted(set(df["condition"]) - {"11", "10", "01", "00"})
    if bad:
        raise ValueError(f"알 수 없는 condition 값: {bad}")
    df["input_mode"] = df["input_mode"].astype(str).str.strip().map(lambda m: MODE_ALIASES.get(m.lower(), m.upper()))
    bad = sorted(set(df["input_mode"]) - {"MM", "TO", "IO"})
    if bad:
        raise ValueError(f"알 수 없는 input_mode 값: {bad}")
    if "score" not in df and "pred" not in df:
        raise ValueError("score 또는 pred 열이 필요합니다")

    df["y"] = (df["condition"] != "00").astype(int)  # 게시물 단위 정답

    dup = df.duplicated(RUN_KEYS + ["split", "sample_id"])
    if dup.any():
        raise ValueError(f"같은 run에 중복 sample_id {dup.sum()}개: {df.loc[dup, 'sample_id'].head().tolist()}")
    return df


def best_ba_threshold(y, s):
    """val에서 BA 최대 임계값. 동점이면 0에 가까운 값."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    best_t, best_ba = 0.0, -1.0
    for t in cands:
        p = s > t
        ba = (p[y == 1].mean() + 1 - p[y == 0].mean()) / 2
        if ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and abs(t) < abs(best_t)):
            best_t, best_ba = float(t), float(ba)
    return best_t


def has_values(df, col):
    return col in df and df[col].notna().any()


def choose_threshold(test, val, val_mm):
    if has_values(test, "pred"):
        return np.nan, "pred"
    if has_values(test, "threshold"):
        vals = test["threshold"].dropna().unique()
        if len(vals) != 1:
            raise ValueError(f"한 run 안에 threshold 값이 여러 개: {vals}")
        return float(vals[0]), "given"
    for source, v in (("val", val), ("val_MM", val_mm)):
        if v is not None and has_values(v, "score"):
            t = best_ba_threshold(v["y"], v["score"])
            if t is not None:
                return t, source
    return 0.0, "default0"


def run_metrics(test, thr, thr_source):
    y = test["y"].to_numpy()
    if thr_source == "pred":
        p = test["pred"].astype(int).to_numpy()
    else:
        p = (test["score"].astype(float).to_numpy() > thr).astype(int)
    both = len(set(y.tolist())) == 2
    out = {"n": len(test), "threshold": thr, "thr_source": thr_source}
    for c in ("11", "10", "01", "00"):
        out[f"n_{c}"] = int((test["condition"] == c).sum())
    # 두 클래스가 모두 있을 때만 전체 지표를 낸다 (한쪽만 있으면 Acc가 곧 탐지율이 되어 오해를 부른다)
    out["acc"] = float((p == y).mean()) if both else np.nan
    out["ba"] = float((p[y == 1].mean() + 1 - p[y == 0].mean()) / 2) if both else np.nan
    out["macro_f1"] = float(f1_score(y, p, average="macro")) if both else np.nan
    out["auroc"] = (float(roc_auc_score(y, test["score"].astype(float))) if both and has_values(test, "score")
                    else np.nan)
    neg = test["condition"].to_numpy() == "00"
    out["fpr"] = float(p[neg].mean()) if neg.any() else np.nan
    for c in ("11", "10", "01"):
        m = test["condition"].to_numpy() == c
        out[f"det_{c}"] = float(p[m].mean()) if m.any() else np.nan
    return out


def evaluate(df):
    rows = []
    test_all, val_all = df[df["split"] == "test"], df[df["split"] == "val"]
    for key, test in test_all.groupby(RUN_KEYS, sort=True):
        k = dict(zip(RUN_KEYS, key))
        same = lambda d: d[(d["model"] == k["model"]) & (d["setting"] == k["setting"]) & (d["seed"] == k["seed"])]
        val_run = same(val_all)
        val = val_run[val_run["input_mode"] == k["input_mode"]]
        val_mm = val_run[val_run["input_mode"] == "MM"]
        thr, src = choose_threshold(test, val if len(val) else None, val_mm if len(val_mm) else None)
        rows.append({**k, **run_metrics(test, thr, src)})
    return pd.DataFrame(rows)


def summarize(per_run):
    out = []
    for key, g in per_run.groupby(["model", "setting", "input_mode"], sort=False):
        row = dict(zip(["model", "setting", "input_mode"], key))
        row["n_seeds"] = len(g)
        row["thr_source"] = "/".join(sorted(g["thr_source"].unique()))
        for m in METRICS:
            v = g[m].dropna()
            row[m] = v.mean() if len(v) else np.nan
            row[f"{m}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def fmt(mean, std, pct):
    if pd.isna(mean):
        return "-"
    s = f"{mean * 100:.1f}" if pct else f"{mean:.3f}"
    if not pd.isna(std):
        s += f"±{std * 100:.1f}" if pct else f"±{std:.3f}"
    return s


def to_markdown(summary, pct=True):
    names = {"acc": "Acc", "ba": "BA", "auroc": "AUROC", "macro_f1": "Macro-F1", "fpr": "FPR@00",
             "det_11": "Det@11", "det_10": "Det@10", "det_01": "Det@01"}
    head = ["Model", "Setting", "Input", "seeds"] + [names[m] for m in METRICS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        cells = [r["model"], r["setting"], r["input_mode"], str(r["n_seeds"])]
        cells += [fmt(r[m], r[f"{m}_std"], pct) for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    unit = "%" if pct else "비율"
    lines += ["", f"값은 {unit}, seed가 2개 이상이면 평균±표준편차. 정답은 게시물 단위(조건 11·10·01=혐오, 00=비혐오).",
              "Det@c는 조건 c의 혐오 판정 비율, FPR@00은 비혐오 게시물의 혐오 판정 비율."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True, help="공통 형식 CSV (glob 가능)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", action="store_true", help="표를 %% 대신 0~1 비율로")
    args = ap.parse_args()

    paths = sorted({p for pat in args.pred for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"파일 없음: {args.pred}")
    df = load(paths)
    per_run = evaluate(df)
    summary = summarize(per_run)

    os.makedirs(args.out, exist_ok=True)
    per_run.to_csv(os.path.join(args.out, "per_run.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False, encoding="utf-8-sig")
    md = to_markdown(summary, pct=not args.ratio)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
