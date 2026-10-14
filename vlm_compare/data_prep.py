"""
image_dataset/{train,test}/metadata.csv -> data/{train,val,test}.jsonl

네 가지 케이스를 모든 split에 만든다 (label 값은 common.py 참고):
    유유 hate                 : 원본 hate 행 그대로
    무무 non_hate             : 원본 non_hate 행 그대로
    유무 image_hate_text_safe : hate 이미지 + non_hate 텍스트 (서로 다른 행끼리 크로스 페어링)
    무유 text_hate_image_safe : non_hate 이미지 + hate 텍스트 (서로 다른 행끼리 크로스 페어링)

split 원칙
    - validation은 원본 train 행에서 group_id 단위 + 라벨 층화로 먼저 떼어낸 뒤,
      train / val / test 각각의 "안에서만" 크로스 페어링한다.
      -> 같은 이미지나 텍스트가 train과 val/test에 동시에 들어가는 누수가 없다.
    - val을 여기서 파일로 고정해두므로 모든 모델이 완전히 같은 train/val/test를 쓴다.

모델 비교 실험용 "기본 코드"라서 --augment(은어/반례 증강)는 넣지 않았다.

Run:
    python data_prep.py --dataset-root ../image_dataset --out-dir data
"""
import argparse
import csv
import json
import os
import random
import unicodedata
from collections import Counter

from common import ALL_CASES, CASE_NAME_KO, LABEL_TO_TARGET, PROMPT_WITH_TEXT, resolve_dataset_root


def normalize_text(text):
    """NFKC 정규화 + 앞뒤 공백 제거 + 연속 공백 축소만. 은어/자모/반복 글자는 보존."""
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.strip().split())


def make_record(row_id, image_path, label, source_text_raw, image_source="unknown",
                hate_categories="", group_id=None):
    text = normalize_text(source_text_raw)
    return {
        "id": row_id,
        "image_path": image_path,
        "label": label,
        "source_text": text,
        "source_text_raw": source_text_raw,
        "image_source": image_source,
        "hate_categories": hate_categories,
        "group_id": group_id if group_id is not None else row_id,
        "prompt": PROMPT_WITH_TEXT.format(source_text=text),
        "target": LABEL_TO_TARGET[label],
    }


def read_base_records(dataset_root, split):
    """원본 metadata.csv에서 hate(유유)/non_hate(무무) 행만 읽는다."""
    csv_path = os.path.join(dataset_root, split, "metadata.csv")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records, n_missing = [], 0
    for row in rows:
        label = (row.get("label") or "").strip()
        if label not in ("hate", "non_hate"):
            continue
        if not os.path.exists(os.path.join(dataset_root, row["image_path"])):
            n_missing += 1
            print(f"[경고] 이미지 없음, 제외: {row['id']} ({row['image_path']})")
            continue
        records.append(make_record(
            row["id"], row["image_path"], label, row.get("source_text", ""),
            image_source=row.get("image_source", "unknown") or "unknown",
            hate_categories=row.get("hate_categories", "") or "",
            group_id=row.get("group_id") or row["id"],
        ))
    if n_missing:
        print(f"[{split}] 이미지 누락으로 제외: {n_missing}개")
    return records


def split_train_val(records, val_frac, rng):
    """group_id 단위 + 라벨 층화로 validation 분리."""
    groups = {}
    for r in records:
        groups.setdefault(r["group_id"], []).append(r)
    group_label = {g: Counter(r["label"] for r in rs).most_common(1)[0][0] for g, rs in groups.items()}

    val_groups = set()
    for label in sorted(set(group_label.values())):
        gids = sorted(g for g, l in group_label.items() if l == label)
        rng.shuffle(gids)
        val_groups.update(gids[:max(1, round(len(gids) * val_frac))])

    train, val = [], []
    for g, rs in groups.items():
        (val if g in val_groups else train).extend(rs)
    return train, val


def cross_pairs(records, rng, case, max_pairs=None):
    """크로스 페어링으로 유무/무유 케이스 생성 (같은 split 안에서만).
    case="image_hate_text_safe": hate 이미지 + non_hate 텍스트
    case="text_hate_image_safe": non_hate 이미지 + hate 텍스트"""
    if case == "image_hate_text_safe":
        img_label, txt_label = "hate", "non_hate"
    elif case == "text_hate_image_safe":
        img_label, txt_label = "non_hate", "hate"
    else:
        raise ValueError(case)

    images = [r for r in records if r["label"] == img_label]
    texts = [r for r in records if r["label"] == txt_label and r["source_text"]]
    images, texts = images[:], texts[:]
    rng.shuffle(images)
    rng.shuffle(texts)

    n = min(len(images), len(texts))
    if max_pairs is not None:
        n = min(n, max_pairs)
    out = []
    for i in range(n):
        img_r, txt_r = images[i], texts[i]
        out.append(make_record(
            f"{case}_{i:04d}_{img_r['id']}_{txt_r['id']}",
            img_r["image_path"], case, txt_r["source_text_raw"],
            image_source=img_r["image_source"],
            hate_categories=txt_r["hate_categories"] if case == "text_hate_image_safe" else img_r["hate_categories"],
            group_id=f"pair_{img_r['group_id']}_{txt_r['group_id']}",
        ))
    return out


def build_split(base_records, rng, max_pairs):
    records = list(base_records)
    records += cross_pairs(base_records, rng, "image_hate_text_safe", max_pairs)
    records += cross_pairs(base_records, rng, "text_hate_image_safe", max_pairs)
    rng.shuffle(records)
    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(records, name):
    c = Counter(r["label"] for r in records)
    parts = ", ".join(f"{CASE_NAME_KO[k]}={c.get(k, 0)}" for k in ALL_CASES)
    n_pos = sum(v for k, v in c.items() if LABEL_TO_TARGET[k] == "혐오")
    print(f"[{name}] 총 {len(records)}개 | {parts} | 혐오:비혐오 = {n_pos}:{len(records) - n_pos}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-pairs-per-case", type=int, default=None,
                    help="유무/무유를 각각 최대 몇 개 만들지. 기본은 가능한 최대(min(hate 수, non_hate 수)).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.dataset_root = resolve_dataset_root(args.dataset_root)
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_base = read_base_records(args.dataset_root, "train")
    test_base = read_base_records(args.dataset_root, "test")
    train_base, val_base = split_train_val(train_base, args.val_frac, rng)

    splits = {
        "train": build_split(train_base, rng, args.max_pairs_per_case),
        "val": build_split(val_base, rng, args.max_pairs_per_case),
        "test": build_split(test_base, rng, args.max_pairs_per_case),
    }
    for name, recs in splits.items():
        write_jsonl(recs, os.path.join(args.out_dir, f"{name}.jsonl"))
        report(recs, name)

    # 누수 점검: 이미지 경로 / 원문 텍스트가 train과 val/test에 겹치는지
    def keys(recs):
        return {r["image_path"] for r in recs}, {r["source_text"] for r in recs if r["source_text"]}
    tr_img, tr_txt = keys(splits["train"])
    for other in ("val", "test"):
        o_img, o_txt = keys(splits[other])
        print(f"[누수 점검] train∩{other}: 이미지 {len(tr_img & o_img)}개, 텍스트 {len(tr_txt & o_txt)}개 겹침")


if __name__ == "__main__":
    main()
