"""
자연도 반영 결합의 후보 채점 (3.7.4 첫 단계)

같은 split 안에서 이미지마다 후보 텍스트 40개를 무작위로 뽑아 seed 1 모델로 자연도 점수를 매긴다.
    조건 10: 유해 게시물의 이미지 x 무해 게시물의 텍스트
    조건 01: 무해 게시물의 이미지 x 유해 게시물의 텍스트
train과 test를 따로 채점하므로 split 간 누수는 없다.

    python score_candidates.py --adapter-dir ../main_experiment/analysis/outputs_v2/seed1/lora-adapter \
        --dataset-root ../image_dataset --out-dir candidates

출력: candidates/candidates_{split}.csv (split, case, image_id, text_id, nat_score)
중간에 끊기면 같은 명령으로 이어서 채점한다. 후보 추출은 --candidate-seed로 고정한다.
"""
import argparse
import os
import random

import pandas as pd

from naturalness_lib import CROSS_CASES, NaturalnessScorer, natural_records

COLS = ["split", "case", "image_id", "text_id", "nat_score"]


def candidate_list(records, n_candidates, rng):
    """[(case, image_record, text_record), ...] 이미지 순서와 후보 추출이 seed로 고정된다."""
    out = []
    for case, (img_label, txt_label) in CROSS_CASES.items():
        images = sorted((r for r in records if r["label"] == img_label), key=lambda r: r["id"])
        texts = sorted((r for r in records if r["label"] == txt_label), key=lambda r: r["id"])
        k = min(n_candidates, len(texts))
        for img in images:
            for txt in rng.sample(texts, k):
                out.append((case, img, txt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True, help="후보 채점에 쓰는 모델 (논문: seed 1)")
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--out-dir", default="candidates")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--n-candidates", type=int, default=40)
    ap.add_argument("--candidate-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="점검용: split마다 앞에서 N쌍만")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scorer = NaturalnessScorer(args.adapter_dir, args.dataset_root)
    for split in args.splits:
        rng = random.Random(f"{args.candidate_seed}-{split}")
        todo = candidate_list(natural_records(args.dataset_root, split), args.n_candidates, rng)
        todo = todo[:args.limit] if args.limit else todo
        out = os.path.join(args.out_dir, f"candidates_{split}.csv")
        rows, done = [], set()
        if os.path.exists(out):
            prev = pd.read_csv(out, dtype={"image_id": str, "text_id": str})
            rows = prev.to_dict("records")
            done = set(zip(prev["case"], prev["image_id"], prev["text_id"]))
            print(f"[{split}] 이어서 채점: 이미 {len(done)}쌍")
        for i, (case, img, txt) in enumerate(todo):
            if (case, img["id"], txt["id"]) in done:
                continue
            rows.append({"split": split, "case": case, "image_id": img["id"], "text_id": txt["id"],
                         "nat_score": scorer.score(img["image_path"], txt["source_text"])})
            if (i + 1) % 500 == 0:
                pd.DataFrame(rows, columns=COLS).to_csv(out, index=False, encoding="utf-8")
                print(f"  [{split}] {i + 1}/{len(todo)}")
        pd.DataFrame(rows, columns=COLS).to_csv(out, index=False, encoding="utf-8")
        print(f"[{split}] {len(rows)}쌍 -> {out}")


if __name__ == "__main__":
    main()
