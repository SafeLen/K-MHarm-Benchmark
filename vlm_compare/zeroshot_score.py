"""
Zero-shot 채점: 학습(LoRA) 없이 원래 모델로 test(와 선택적으로 val)를 세 가지 입력 방식으로 채점한다.

    python zeroshot_score.py --model varco
    python zeroshot_score.py --model internvl3 --val-jsonl data/val.jsonl   # val도 채점 (임계값 선택용)

입력 방식 (논문 3.5절과 동일)
    multimodal  원래 이미지 + 게시글 포함 프롬프트
    text_only   회색 이미지(384x384, RGB 128) + 게시글 포함 프롬프트   (이미지 토큰 위치는 유지)
    image_only  원래 이미지 + 게시글을 뺀 프롬프트

점수: score = log P("혐오") - log P("비혐오")  (teacher forcing, 주 실험과 같은 함수)
입력 방식별 정답 (논문 3.5.2절)
    multimodal = OR 규칙 정답, text_only = H_T, image_only = H_I

출력: results_zeroshot/<model>.csv  (중간에 끊겨도 다시 실행하면 이어서 채점)
    model, split, sample_id, condition, input_mode, score, y_post, y_input, H_I, H_T, logp_hate, logp_nonhate
    - 앞 4개(sample_id, condition, input_mode, score)가 B 공통 스크립트 입력 형식
    - y_post  = 게시물 단위 정답(OR 규칙, 입력 방식과 무관)  <- 메인 표 기준
    - y_input = 입력 방식별 정답(논문 3.5.2, 참고용)
"""
import argparse
import json
import os

import pandas as pd
import torch
from PIL import Image

import image_prep
from common import PROMPT_WITH_TEXT, resolve_dataset_root
from models import get_adapter, hate_score

# 게시글을 뺀 이미지 판단 프롬프트 (PROMPT_WITH_TEXT에서 게시글 부분만 제거)
PROMPT_IMAGE_ONLY = (
    "다음 이미지를 보고 혐오 표현(혐오, 차별, 비하 등)을 담고 있는지 판단하세요. "
    "반드시 '혐오' 또는 '비혐오' 중 하나의 단어로만 답하세요."
)

# label 값 -> (조건, H_I, H_T). 주 실험 데이터의 label 이름이 다르면 여기에 추가하면 된다.
LABEL_MAP = {
    "hate": ("11", 1, 1), "cond11": ("11", 1, 1), "11": ("11", 1, 1),
    "image_hate_text_safe": ("10", 1, 0), "cond10": ("10", 1, 0), "10": ("10", 1, 0),
    "text_hate_image_safe": ("01", 0, 1), "cond01": ("01", 0, 1), "01": ("01", 0, 1),
    "non_hate": ("non", 0, 0), "nonhate": ("non", 0, 0), "00": ("non", 0, 0),
}
INPUTS = ["multimodal", "text_only", "image_only"]


def y_for_input(inp, h_i, h_t):
    if inp == "multimodal":
        return int(h_i or h_t)
    return h_t if inp == "text_only" else h_i


def load_jsonl(path, split, limit=None):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = recs[:limit] if limit else recs
    for r in recs:
        r["_split"] = split
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-jsonl", default="data/test.jsonl")
    ap.add_argument("--val-jsonl", default=None, help="주면 val도 채점 -> zeroshot_metrics.py가 val에서 임계값 선택")
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--out-dir", default="results_zeroshot")
    ap.add_argument("--inputs", default=",".join(INPUTS))
    ap.add_argument("--limit", type=int, default=None, help="시험용: 앞에서 N개만")
    args = ap.parse_args()

    args.dataset_root = resolve_dataset_root(args.dataset_root)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}.csv")
    inputs = [x.strip() for x in args.inputs.split(",")]

    recs = load_jsonl(args.test_jsonl, "test", args.limit)
    if args.val_jsonl:
        recs += load_jsonl(args.val_jsonl, "val", args.limit)

    done = set()
    rows = []
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, dtype={"condition": str, "sample_id": str})
        rows = prev.to_dict("records")
        done = set(zip(prev.split, prev.input_mode, prev.sample_id.astype(str)))
        print(f"이어서 채점: 이미 {len(done)}건 완료")

    unknown = sorted({r["label"] for r in recs if r["label"] not in LABEL_MAP})
    assert not unknown, f"LABEL_MAP에 없는 label: {unknown} -> zeroshot_score.py의 LABEL_MAP에 추가하세요."

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = get_adapter(args.model, dev)
    model = adapter.load().eval()
    gray = Image.new("RGB", (image_prep.IMAGE_SIZE, image_prep.IMAGE_SIZE), (128, 128, 128))

    n_new = 0
    for i, r in enumerate(recs):
        todo = [inp for inp in inputs if (r["_split"], inp, str(r["id"])) not in done]
        if not todo:
            continue
        img = image_prep.prepare_image(os.path.join(args.dataset_root, r["image_path"]), "test", sample_id=r["id"])
        if img is None:
            continue
        cond, h_i, h_t = LABEL_MAP[r["label"]]
        prompt_text = r.get("prompt") or PROMPT_WITH_TEXT.format(source_text=r.get("source_text", ""))
        for inp in todo:
            if inp == "multimodal":
                s, lh, ln = hate_score(adapter, model, img, prompt_text)
            elif inp == "text_only":
                s, lh, ln = hate_score(adapter, model, gray, prompt_text)
            else:
                s, lh, ln = hate_score(adapter, model, img, PROMPT_IMAGE_ONLY)
            rows.append({"model": args.model, "split": r["_split"], "sample_id": str(r["id"]),
                         "condition": cond, "input_mode": inp, "score": s,
                         "y_post": int(h_i or h_t), "y_input": y_for_input(inp, h_i, h_t),
                         "H_I": h_i, "H_T": h_t, "logp_hate": lh, "logp_nonhate": ln})
            n_new += 1
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  {i + 1}/{len(recs)}")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[{args.model}] 새로 채점 {n_new}건 -> {out_path}")


if __name__ == "__main__":
    main()
