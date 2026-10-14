"""
UnSmile -> 원문 1,000개 (혐오 500 / 무해 500) -> train 700 / test 300 metadata.csv (논문 3.1)

라벨 재정의 (논문 3.1, text_baseline.py와 동일)
    혐오  : 8개 혐오 범주(여성/가족, 남성, 성소수자, 인종/국적, 연령, 지역, 종교, 기타 혐오) 중 하나 이상이 1
    무해  : clean=1 이고 혐오 범주·악플/욕설이 모두 0
    제외  : 악플/욕설만 있는 문장
원천: UnSmile 공식 train + valid TSV (smilegate-ai/korean_unsmile_dataset)

select : 위 규칙으로 중복을 지운 뒤 클래스마다 500개를 뽑아 원문 단위로 train 350 / test 150으로 나눈다.
         출력은 build_image_dataset.py와 vlm_compare/data_prep.py, main_experiment/shared/data_prep.py가
         읽는 형식 ({split}/metadata.csv: id, label, source_text, image_path, hate_categories)이다.
verify : 공개한 metadata.csv가 위 규칙을 지키는지 점검한다 (UnSmile에 있는지, 재정의 라벨·범주가 같은지,
         개수, split 간 텍스트 중복).

참고: 논문에 쓴 1,000개는 원래 추출에서 정해진 것이고, 당시 id와 난수 상태는 남아 있지 않다.
같은 seed로 select를 돌려도 같은 1,000개가 나오지는 않는다. 논문 결과를 재현하려면 공개한
metadata.csv를 쓰고, 그 데이터가 위 규칙을 따르는지는 verify로 확인한다.

Usage
    python build_unsmile_subset.py select --unsmile-dir unsmile --out-dir ../image_dataset --seed 42
    python build_unsmile_subset.py verify --unsmile-dir unsmile --dataset-root ../image_dataset
"""
import argparse
import hashlib
import os
import random
import re
import sys

import pandas as pd

HATE_CATS = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오"]
SPLIT_SIZES = {"train": 350, "test": 150}  # 클래스당


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def load_pool(unsmile_dir):
    """UnSmile train+valid에 라벨 재정의를 적용한다. 같은 문장은 하나만 남긴다
    (같은 문장이 서로 다른 라벨로 나오면 모호하므로 둘 다 뺀다)."""
    parts = []
    for name in ("train", "valid"):
        df = pd.read_csv(os.path.join(unsmile_dir, f"unsmile_{name}_v1.0.tsv"), sep="\t")
        parts.append(df.assign(unsmile_split=name))
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["문장"].map(norm)
    hate = df[HATE_CATS].sum(axis=1) > 0
    clean = (df["clean"] == 1) & (df[HATE_CATS + ["악플/욕설"]].sum(axis=1) == 0)
    df = df[(hate | clean) & (df["text"] != "")].copy()
    df["label"] = (df[HATE_CATS].sum(axis=1) > 0).map({True: "hate", False: "non_hate"})
    df["hate_categories"] = df[HATE_CATS].apply(lambda r: "|".join(c for c in HATE_CATS if r[c] == 1), axis=1)
    ambiguous = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(ambiguous[ambiguous > 1].index)]
    return df.drop_duplicates("text").reset_index(drop=True)


def make_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def select(args):
    pool = load_pool(args.unsmile_dir)
    rng = random.Random(args.seed)
    n_per_class = sum(SPLIT_SIZES.values())
    rows = {s: [] for s in SPLIT_SIZES}
    for label in ("hate", "non_hate"):
        idx = pool.index[pool["label"] == label].tolist()
        if len(idx) < n_per_class:
            raise SystemExit(f"{label} 후보가 {len(idx)}개뿐입니다 (필요 {n_per_class})")
        rng.shuffle(idx)
        picked = idx[:n_per_class]
        start = 0
        for split, n in SPLIT_SIZES.items():
            for i in picked[start:start + n]:
                r = pool.loc[i]
                sid = make_id(r["text"])
                rows[split].append({"id": sid, "label": label, "source_text": r["문장"],
                                    "image_path": f"{split}/{label}/{sid}.png",
                                    "hate_categories": r["hate_categories"]})
            start += n
    for split, rs in rows.items():
        d = os.path.join(args.out_dir, split)
        os.makedirs(d, exist_ok=True)
        df = pd.DataFrame(rs).sample(frac=1, random_state=args.seed)
        df.to_csv(os.path.join(d, "metadata.csv"), index=False, encoding="utf-8")
        print(f"[{split}] {len(df)}개 ({df.label.value_counts().to_dict()}) -> {d}/metadata.csv")
    print("다음 단계: build_image_dataset.py generate 로 이미지를 만든다.")


def verify(args):
    pool = load_pool(args.unsmile_dir).set_index("text")
    problems, seen = [], {}
    for split, n in SPLIT_SIZES.items():
        meta = pd.read_csv(os.path.join(args.dataset_root, split, "metadata.csv"), dtype=str).fillna("")
        counts = meta["label"].value_counts().to_dict()
        if counts != {"hate": n, "non_hate": n}:
            problems.append(f"[{split}] 클래스 개수 {counts} (기대값 각 {n})")
        for _, r in meta.iterrows():
            t = norm(r["source_text"])
            if t in seen:
                problems.append(f"[{split}] {r['id']}: {seen[t]}와 같은 텍스트")
            seen[t] = f"{split}/{r['id']}"
            if t not in pool.index:
                problems.append(f"[{split}] {r['id']}: UnSmile(재정의 후)에 없는 텍스트")
                continue
            p = pool.loc[t]
            if p["label"] != r["label"]:
                problems.append(f"[{split}] {r['id']}: 라벨 {r['label']} != 재정의 {p['label']}")
            if p["hate_categories"] != r["hate_categories"]:
                problems.append(f"[{split}] {r['id']}: 범주 {r['hate_categories']!r} != {p['hate_categories']!r}")
        print(f"[{split}] {len(meta)}개 점검 {counts}")
    if problems:
        print(f"문제 {len(problems)}건:")
        print("\n".join(problems[:50]))
        sys.exit(1)
    print("모든 표본이 UnSmile 라벨 재정의와 일치하고, split 간 텍스트 중복이 없습니다.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--unsmile-dir", default="unsmile")
    s.add_argument("--out-dir", default="../image_dataset")
    s.add_argument("--seed", type=int, default=42)
    v = sub.add_parser("verify")
    v.add_argument("--unsmile-dir", default="unsmile")
    v.add_argument("--dataset-root", default="../image_dataset")
    args = ap.parse_args()
    select(args) if args.cmd == "select" else verify(args)


if __name__ == "__main__":
    main()
