"""
각 seed 모델이 자신의 평가셋 600건(조건 11, 10, 01, 비혐오 각 150건)에 자연도 점수를 매긴다 (3.7.2).

    python score_naturalness.py --adapter-dir ../main_experiment/analysis/outputs_v2/seed1/lora-adapter \
        --jsonl ../main_experiment/analysis/data_v2/seed1/test.jsonl --dataset-root ../image_dataset \
        --out nat/nat_seed1.csv

출력 열: id, label, image_group_id, text_group_id, nat_score
중간에 끊기면 같은 명령으로 이어서 채점한다.
"""
import argparse
import json
import os

import pandas as pd

from naturalness_lib import NaturalnessScorer

COLS = ["id", "label", "image_group_id", "text_group_id", "nat_score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="점검용: 앞에서 N건만")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    records = records[:args.limit] if args.limit else records
    rows, done = [], set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out, dtype={"id": str})
        rows, done = prev.to_dict("records"), set(prev["id"])
        print(f"이어서 채점: 이미 {len(done)}건")

    scorer = NaturalnessScorer(args.adapter_dir, args.dataset_root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for i, r in enumerate(records):
        if str(r["id"]) in done:
            continue
        rows.append({"id": str(r["id"]), "label": r["label"],
                     "image_group_id": r.get("image_group_id", r["id"]),
                     "text_group_id": r.get("text_group_id", r["id"]),
                     "nat_score": scorer.score(r["image_path"], r["source_text"])})
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows, columns=COLS).to_csv(args.out, index=False, encoding="utf-8")
            print(f"  {i + 1}/{len(records)}")
    pd.DataFrame(rows, columns=COLS).to_csv(args.out, index=False, encoding="utf-8")
    print(f"{len(rows)}건 -> {args.out}")


if __name__ == "__main__":
    main()
