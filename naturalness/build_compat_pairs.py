"""
자연도 반영 결합 데이터 만들기 (3.7.4 둘째 단계)

seed마다 이미지별 후보 40개 중 20개를 무작위로 다시 골라, 각 이미지와 텍스트가 정확히 한 번씩
쓰이도록 자연도 점수 합이 최대가 되는 배정(선형 할당)으로 조건 10·01의 짝을 만든다. 원본 표본
(조건 11·비혐오)은 그대로 두고, 레코드 형식·id 규칙·match_cluster_id는 data_prep.py와 같다.
후보 밖의 짝이 배정되면(완전 매칭이 불가능할 때) 그 수를 pairing_summary.json의 n_fallback에 남긴다.
논문 실험에서는 5개 seed 모두 0건이었다.

    python build_compat_pairs.py --dataset-root ../image_dataset --candidates-dir candidates \
        --out-dir data_compat --seeds 1 2 3 4 5

출력: data_compat/seed<N>/{train,test}.jsonl, {train,test}_image_only.jsonl, pairing_summary.json
이후 학습·평가는 main_experiment와 같은 설정으로 run_naturalness.sh가 수행한다.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from naturalness_lib import CROSS_CASES, natural_records

import data_prep  # naturalness_lib가 main_experiment/shared를 sys.path에 넣는다

NOT_CANDIDATE = 1e6


def assign_pairs(cand, images, texts, n_select, rng):
    """cand: 한 split·한 case의 후보 점수 (image_id, text_id, nat_score).
    반환: [(image_record, text_record, nat_score or None)], 후보 밖 배정 수."""
    img_idx = {r["id"]: i for i, r in enumerate(images)}
    txt_idx = {r["id"]: j for j, r in enumerate(texts)}
    cost = np.full((len(images), len(texts)), NOT_CANDIDATE)
    for image_id, g in cand.groupby("image_id", sort=True):
        rows = sorted(zip(g["text_id"], g["nat_score"]))
        for text_id, score in rng.sample(rows, min(n_select, len(rows))):
            cost[img_idx[image_id], txt_idx[text_id]] = -score
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs, n_fallback = [], 0
    for i, j in zip(row_ind, col_ind):
        is_cand = cost[i, j] < NOT_CANDIDATE
        n_fallback += not is_cand
        pairs.append((images[i], texts[j], -cost[i, j] if is_cand else None))
    return pairs, n_fallback


def build_split(dataset_root, split, cand_split, seed, n_select):
    rng = random.Random(f"{seed}-{split}")
    base = natural_records(dataset_root, split)
    extras, summary = [], {}
    for case, (img_label, txt_label) in CROSS_CASES.items():
        images = sorted((r for r in base if r["label"] == img_label), key=lambda r: r["id"])
        texts = sorted((r for r in base if r["label"] == txt_label), key=lambda r: r["id"])
        cand = cand_split[cand_split["case"] == case]
        missing = {r["id"] for r in images} - set(cand["image_id"])
        if missing:
            raise ValueError(f"[{split}/{case}] 후보 점수가 없는 이미지 {len(missing)}개")
        pairs, n_fallback = assign_pairs(cand, images, texts, n_select, rng)
        rng.shuffle(pairs)
        for i, (img, txt, score) in enumerate(pairs):
            rec = data_prep.make_record(
                f"{case}_{i:04d}_{txt['id']}", img["image_path"], case, txt["source_text_raw"],
                include_text=True, mask_names=False, hate_categories="",
                image_source=img.get("image_source", "unknown"),
                image_group_id=img["image_group_id"], text_group_id=txt["text_group_id"])
            rec["nat_score"] = score
            extras.append(rec)
        chosen = [s for _, _, s in pairs if s is not None]
        summary[case] = {"n_pairs": len(pairs), "n_fallback": int(n_fallback),
                         "selected_mean_nat": float(np.mean(chosen)) if chosen else None,
                         "candidate_mean_nat": float(cand["nat_score"].mean())}
    records = base + extras
    rng.shuffle(records)
    data_prep.assign_match_clusters(records)
    return records, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--candidates-dir", default="candidates")
    ap.add_argument("--out-dir", default="data_compat")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-select", type=int, default=20)
    args = ap.parse_args()

    cands = {s: pd.read_csv(os.path.join(args.candidates_dir, f"candidates_{s}.csv"),
                            dtype={"image_id": str, "text_id": str}) for s in ("train", "test")}
    for seed in args.seeds:
        out = os.path.join(args.out_dir, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        summary, split_records = {}, {}
        for split in ("train", "test"):
            records, summary[split] = build_split(args.dataset_root, split, cands[split], seed, args.n_select)
            split_records[split] = records
            data_prep.write_jsonl(records, os.path.join(out, f"{split}.jsonl"))
            data_prep.write_jsonl([data_prep.make_image_only_record(r) for r in records],
                                  os.path.join(out, f"{split}_image_only.jsonl"))
        data_prep.report_group_overlap(split_records["train"], split_records["test"])
        with open(os.path.join(out, "pairing_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        msg = ", ".join(f"{sp}/{c}: 선택 {v['selected_mean_nat']:.2f} vs 후보 {v['candidate_mean_nat']:.2f}"
                        f" (fallback {v['n_fallback']})" for sp, d in summary.items() for c, v in d.items())
        print(f"[seed{seed}] {msg}")


if __name__ == "__main__":
    main()
