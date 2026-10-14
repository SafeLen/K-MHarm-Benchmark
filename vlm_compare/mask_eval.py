"""
모달리티 가리기 평가. 학습된 LoRA 어댑터를 그대로 쓰고 입력만 가려서 test를 다시 채점한다.
  --mask image : 이미지를 회색 384x384로 교체, 텍스트는 그대로
  --mask text  : 이미지는 그대로, 게시글 텍스트를 빈 문자열로 (프롬프트 형식은 동일)
임계값은 원래 평가와 같은 값(run_info.json의 selected_threshold)을 그대로 쓴다.
"""
import argparse, json, math, os
import pandas as pd
import torch
from PIL import Image

import image_prep
from common import PROMPT_WITH_TEXT, is_hate
from metrics import print_rows, save_all
from models import get_adapter, hate_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["varco", "hyperclovax", "bllossom", "ax", "qwen3vl"])
ap.add_argument("--mask", required=True, choices=["image", "text"])
ap.add_argument("--dataset-root", required=True)
ap.add_argument("--adapter-dir", default=None)
ap.add_argument("--test-jsonl", default="data/test.jsonl")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

adapter_dir = args.adapter_dir or os.path.join("outputs", args.model)
out_dir = os.path.join(adapter_dir, f"eval_mask_{args.mask}")
os.makedirs(out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

run_info = json.load(open(os.path.join(adapter_dir, "run_info.json"), encoding="utf-8"))
threshold = run_info["selected_threshold"]
print(f"[{args.model}] mask={args.mask} | 임계값(원래 평가와 동일) {threshold:.4f}")

adapter = get_adapter(args.model, device)
base = adapter.load()
from peft import PeftModel
model = PeftModel.from_pretrained(base, adapter_dir).to(device)
model.eval()

gray = Image.new("RGB", (image_prep.IMAGE_SIZE, image_prep.IMAGE_SIZE), color=(128, 128, 128))
records = [json.loads(l) for l in open(args.test_jsonl, encoding="utf-8")]
if args.limit:
    records = records[:args.limit]

rows = []
for i, r in enumerate(records):
    if args.mask == "image":
        img, prompt = gray, r["prompt"]
    else:
        img = image_prep.prepare_image(os.path.join(args.dataset_root, r["image_path"]), "test", sample_id=r["id"])
        if img is None:
            continue
        prompt = PROMPT_WITH_TEXT.format(source_text="")
    score, lp_h, lp_n = hate_score(adapter, model, img, prompt)
    rows.append({"id": r["id"], "label": r["label"], "target": r["target"], "y_true": is_hate(r),
                 "score": score, "prob_hate": 1 / (1 + math.exp(-max(min(score, 50.0), -50.0))),
                 "logp_hate": lp_h, "logp_nonhate": lp_n, "y_pred": int(score > threshold),
                 "source_text": r.get("source_text", "")})
    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{len(records)}")

if not rows:
    raise SystemExit("[중단] 채점된 표본이 없습니다. --dataset-root를 확인하세요.")
pred = pd.DataFrame(rows)
pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")
metric_rows = save_all(pred, out_dir, {"model_key": args.model, "model_display": adapter.display,
                                       "mask": args.mask, "threshold": threshold})
print(f"\n[{adapter.display}] mask={args.mask}")
print_rows(metric_rows)
print(f"저장: {out_dir}")
