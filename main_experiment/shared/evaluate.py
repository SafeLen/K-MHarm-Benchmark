"""
Evaluate a VARCO-VISION-2.0-1.7B LoRA adapter on the test set.

이 파일은 박나현/조현서/정미정 세 사람이 공유하는 단일 evaluate.py다(shared/ 아래 1벌만
존재). 모델은 --case-pairs로 만든 세 조건(hate/image_hate_text_safe/text_hate_image_safe)
+ non_hate를 합친 데이터로 딱 한 번만 학습되므로, "각자 자기 조건만 보고 싶을 때"는
--eval-filter로 test.jsonl 중 그 label만 골라서 채점한다.

For each image we score both candidate completions ("혐오" and "비혐오") by
their teacher-forced log-likelihood under the model and pick the higher one.
This is more robust than greedy-decoding + string matching because it isn't
affected by the model wandering off the two-word answer format.

Evaluation can be restricted with --eval-filter:
    - all (default): full test set, accuracy/F1/AUROC as usual.
    - 그 외 임의 값: test.jsonl의 label 필드가 정확히 이 값과 일치하는 것만 채점합니다.
      예: 'hate'(박나현 케이스), 'image_hate_text_safe'(조현서 케이스),
      'text_hate_image_safe'(정미정 케이스). data_prep.py가 생성한 label 값과 맞춰서
      넣으면 어떤 케이스든 걸러서 채점할 수 있습니다. Training/adapter는 영향 없음 --
      채점 대상 test 예시만 필터링합니다. 필터링된 조건이 단일 클래스(모두 유해)면
      F1/AUROC은 정의되지 않으므로 대신 detection rate(recall)를 보고합니다.

--blank-image replaces every test image with a flat gray square, keeping the
text prompt unchanged. Comparing detection rate with/without this flag shows
whether the model is actually reading the image or just guessing off the
text/prior ("멀티모달 기여도 검증" — same idea as ablating image input to
isolate a multimodal model's per-modality contribution). 이 CSV들이
analysis/matched_design_analysis.py와 shared/modality_bias_report.py가 쓰는
Full/Text-only/Image-only 원점수 분포의 원천이다.

임계값(--threshold): 연구 설계서 6장 원칙대로, validation에서 고른 임계값을
test에는 고정해서 그대로 적용해야 한다. --adapter-dir/run_info.json에
train_lora.py가 저장한 selected_threshold가 있으면 자동으로 읽어 쓰고,
--threshold를 명시하면 그 값으로 덮어쓴다. 둘 다 없으면 기존과 동일하게 0.0.

Usage:
    # 박나현 케이스만 채점
    python evaluate.py --dataset-root ../image_dataset --test-jsonl data/test.jsonl \
        --adapter-dir outputs/lora-adapter --out outputs/test_predictions_hate.csv \
        --eval-filter hate

    # image 기여도 검증: 위와 같은 test set을 회색 이미지로 다시 채점 (text-only)
    python evaluate.py ... --blank-image --out outputs/test_predictions_hate_text_only.csv
"""
import argparse
import json
import math
import os

import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from peft import PeftModel

import image_prep

MODEL_ID = "NCSOFT/VARCO-VISION-2.0-1.7B"
HATE_TOKEN = "혐오"
NONHATE_TOKEN = "비혐오"
# Must match train_lora.py's IMAGE_SIZE — training resizes to a single
# anyres tile to fit T4 memory, so eval needs the same preprocessing.
IMAGE_SIZE = image_prep.IMAGE_SIZE


@torch.no_grad()
def last_token_hidden_state(model, processor, image, prompt_text, device, dtype):
    """t-SNE 시각화(viz_lib.plot_tsne_last_token, embedding_tsne_report.py)용: 프롬프트
    마지막 토큰의 last-layer hidden state를 뽑는다. 기존 판정 로직(sequence_logprob/score)
    에는 전혀 관여하지 않는 --dump-hidden-states 전용 부가 경로."""
    enc = processor(text=prompt_text, images=image, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    pixel_values = enc["pixel_values"].to(device)
    if device == "cuda":
        pixel_values = pixel_values.to(dtype)
    kwargs = {"pixel_values": pixel_values}
    if "image_sizes" in enc:
        kwargs["image_sizes"] = enc["image_sizes"].to(device)
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, **kwargs,
    )
    last_hidden = outputs.hidden_states[-1][0, -1, :]  # (hidden_dim,)
    return last_hidden.float().cpu().numpy()


@torch.no_grad()
def sequence_logprob(model, processor, image, prompt_text, target_text, device, dtype):
    full_text = prompt_text + target_text
    prompt_enc = processor(text=prompt_text, images=image, return_tensors="pt")
    full_enc = processor(text=full_text, images=image, return_tensors="pt")

    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    pixel_values = full_enc["pixel_values"].to(device)
    if device == "cuda":
        pixel_values = pixel_values.to(dtype)
    kwargs = {"pixel_values": pixel_values}
    if "image_sizes" in full_enc:
        kwargs["image_sizes"] = full_enc["image_sizes"].to(device)

    prompt_len = prompt_enc["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    # outputs.loss is mean NLL over supervised (non -100) tokens; convert to summed logprob
    n_target_tokens = (labels != -100).sum().item()
    return -outputs.loss.item() * n_target_tokens


def score_to_prob(score):
    """score = logp_hate - logp_nonhate (로그오즈와 동일한 형태) -> [0,1] 확률로 변환.
    soft label과의 JSD/soft accuracy 계산, Agent 라우팅의 확신도 구간(0.4~0.6) 판단에 쓴다."""
    return 1.0 / (1.0 + math.exp(-score))


def jensen_shannon_divergence(p, q, base=2.0):
    """이진 분포 [p_hate, p_nonhate] 사이의 JSD (soft label 비교용, 5-6장/6장)."""
    p = np.array([p, 1 - p])
    q = np.array([q, 1 - q])
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd_nats = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return jsd_nats / math.log(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--test-jsonl", default="data/test.jsonl")
    ap.add_argument("--adapter-dir", default="outputs/lora-adapter")
    ap.add_argument("--out", default="outputs/test_predictions.csv")
    ap.add_argument(
        "--eval-filter", default="all",
        help="'all'(기본값)이면 전체 test. 그 외 값을 주면 test 예시의 label 필드가 정확히 "
        "이 값과 일치하는 것만 채점합니다 -- 예: 'hate', 'image_hate_text_safe', "
        "'text_hate_image_safe'. data_prep.py가 생성한 label 값과 맞춰서 넣으면 어떤 "
        "케이스든 걸러서 채점할 수 있습니다. Training/adapter는 영향 없음 -- 채점 대상 "
        "test 예시만 필터링합니다.",
    )
    ap.add_argument(
        "--blank-image", action="store_true",
        help="Replace every test image with a flat gray square before scoring "
        "(text prompt unchanged). Use to check whether the model actually "
        "relies on the image — a real image-driven model should score worse "
        "here than with the real image.",
    )
    ap.add_argument(
        "--error-analysis-n", type=int, default=10,
        help="How many top FN/FP rows (by |score|) to print and save as "
        "outputs/<out-stem>_fn.csv / _fp.csv. 0 disables.",
    )
    ap.add_argument(
        "--dump-hidden-states", default=None,
        help="지정하면 각 표본의 last-token hidden state를 이 경로(.npz)에 함께 저장한다 "
        "(viz_lib.plot_tsne_last_token/embedding_tsne_report.py용). 기본 동작에는 영향 "
        "없음(opt-in).",
    )
    ap.add_argument(
        "--threshold", type=float, default=None,
        help="hate-score(logp_hate - logp_nonhate) 판정 임계값. 미지정 시 "
        "--adapter-dir/run_info.json의 selected_threshold(validation에서 고른 값)를 "
        "쓰고, 그마저 없으면 0.0. test에는 항상 고정된 값 하나만 적용한다.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16은 VARCO-VISION 순전파 중 오버플로우로 NaN이 나는 경우가 있어 bf16으로 변경.
    # A100은 bf16을 하드웨어 가속으로 지원해서 fp32만큼 안전하면서 속도 손해가 거의 없다.
    infer_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # use_fast=True: train_lora.py와 동일 이유(slow 이미지 프로세서가 anyres 타일링을
    # CPU에서 표본마다 반복해 극단적으로 느려짐, 실측으로 확인).
    processor = AutoProcessor.from_pretrained(args.adapter_dir, use_fast=True)
    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=infer_dtype,
        attn_implementation="sdpa",
    ).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir).to(device)
    model.eval()

    threshold = args.threshold
    if threshold is None:
        run_info_path = os.path.join(args.adapter_dir, "run_info.json")
        threshold = 0.0
        if os.path.exists(run_info_path):
            with open(run_info_path, "r", encoding="utf-8") as f:
                run_info = json.load(f)
            if run_info.get("selected_threshold") is not None:
                threshold = run_info["selected_threshold"]
                print(f"run_info.json에서 validation 임계값을 불러왔습니다: {threshold:.4f}")
    print(f"판정 임계값(고정, validation에서 선택): {threshold:.4f}")

    records = [json.loads(l) for l in open(args.test_jsonl, "r", encoding="utf-8")]
    if args.eval_filter != "all":
        records = [r for r in records if r["label"] == args.eval_filter]
        assert records, (
            f"--eval-filter {args.eval_filter}: {args.test_jsonl}에 해당 label을 가진 예시가 "
            "없습니다 -- data_prep.py를 그 케이스를 생성하는 옵션(--case-pairs)으로 다시 "
            "돌렸는지 확인하세요."
        )
        print(f"--eval-filter {args.eval_filter}: scoring only {len(records)} test examples")

    blank_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
    if args.blank_image:
        print("--blank-image: 모든 test 이미지를 회색 이미지로 대체해서 채점합니다 "
              "(텍스트만으로 예측하는 것과 동일한 효과 — image 기여도 검증용).")

    rows = []
    n_skipped = 0
    hidden_states, hidden_ids, hidden_labels = [], [], []
    for r in records:
        if args.blank_image:
            image = blank_image
        else:
            image = image_prep.prepare_image(
                os.path.join(args.dataset_root, r["image_path"]), "test", sample_id=r["id"],
            )
            if image is None:
                n_skipped += 1
                continue
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": r["prompt"]},
            ]},
        ]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

        logp_hate = sequence_logprob(model, processor, image, prompt_text, HATE_TOKEN, device, infer_dtype)
        logp_nonhate = sequence_logprob(model, processor, image, prompt_text, NONHATE_TOKEN, device, infer_dtype)

        score = logp_hate - logp_nonhate  # higher => more likely "hate"
        prob_hate = score_to_prob(score)
        pred_label = "hate" if score > threshold else "non_hate"

        row = {
            # image_group_id/text_group_id: matched_design_analysis.py가 세 조건(hate/
            # image_hate_text_safe/text_hate_image_safe) 간 크로스 조건 매칭 샘플을
            # 추적하는 데 쓴다(crossed random effects, cluster bootstrap). match_cluster_id
            # 는 train_lora.py의 split_train_val과 동일한 누수 방지 클러스터(둘 중
            # 하나라도 겹치면 같은 클러스터)로, 진단/재현용으로 함께 남긴다.
            "id": r["id"],
            "image_group_id": r.get("image_group_id", r["id"]),
            "text_group_id": r.get("text_group_id", r["id"]),
            "match_cluster_id": r.get("match_cluster_id", r["id"]),
            "true_label": r["label"], "true_target": r["target"], "pred_label": pred_label,
            "source_text": r.get("source_text", ""),
            "logp_hate": logp_hate, "logp_nonhate": logp_nonhate, "score": score,
            "prob_hate": prob_hate,
            # Tier 2 라우팅(설계서 5-2)이 바로 쓸 수 있게 확신도 구간 플래그를 같이 남긴다.
            "low_confidence": 0.4 <= prob_hate <= 0.6,
        }
        if "soft_label" in r and r["soft_label"] is not None:
            row["soft_label"] = r["soft_label"]
        rows.append(row)

        if args.dump_hidden_states:
            hidden_states.append(last_token_hidden_state(model, processor, image, prompt_text, device, infer_dtype))
            hidden_ids.append(r["id"])
            hidden_labels.append(r["label"])
    if args.dump_hidden_states and hidden_states:
        os.makedirs(os.path.dirname(args.dump_hidden_states) or ".", exist_ok=True)
        np.savez(
            args.dump_hidden_states,
            hidden_states=np.stack(hidden_states),
            ids=np.array(hidden_ids),
            labels=np.array(hidden_labels),
        )
        print(f"hidden states saved to {args.dump_hidden_states}")
    if n_skipped:
        print(f"손상 이미지로 채점에서 제외된 표본: {n_skipped}개")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    # true_label(케이스 이름 문자열, 예: "image_hate_text_safe")이 아니라 true_target(실제
    # 정답 토큰, "혐오"/"비혐오")으로 유해 여부를 판정한다. label이 literally "hate"가
    # 아닌 케이스(조현서/정미정 케이스)도 여기서 정확히 유해로 판정되어야 하기 때문.
    y_true = (df["true_target"] == HATE_TOKEN).astype(int)
    y_pred = (df["pred_label"] == "hate").astype(int)

    if y_true.nunique() < 2:
        # single-class eval (예: --eval-filter hate): F1/AUROC are undefined.
        # Report the detection rate (recall) instead: of the genuinely
        # harmful (hate) test cases, how many did the model still flag?
        n = len(df)
        n_detected = int((y_pred == y_true).sum())
        n_missed = n - n_detected
        recall = n_detected / n if n else float("nan")
        print(f"[eval-filter={args.eval_filter}] n={n}")
        print(f"detection rate (recall on hate): {recall:.3f}  ({n_detected}/{n} detected, {n_missed} missed/FN)")
        if n_missed:
            missed_ids = df.loc[y_pred != y_true, "id"].tolist()
            print(f"missed ids: {missed_ids}")
    else:
        acc = accuracy_score(y_true, y_pred)
        f1_hate = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        precision_hate = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_hate = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auroc = roc_auc_score(y_true, df["score"])
        auprc = average_precision_score(y_true, df["prob_hate"])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        print(f"accuracy: {acc:.3f}")
        print(f"macro-F1: {macro_f1:.3f}")
        print(f"hate precision/recall/F1: {precision_hate:.3f} / {recall_hate:.3f} / {f1_hate:.3f}")
        print(f"AUROC: {auroc:.3f}")
        print(f"AUPRC (hate): {auprc:.3f}")
        print(f"confusion matrix [[TN,FP],[FN,TP]]: [[{tn},{fp}],[{fn},{tp}]]")

        # soft label이 있으면(사람 라벨러 간 불일치 반영) soft accuracy/soft macro-F1/JSD도 보고.
        # 설계서 5-6/6장: hard label(다수결) 대신 soft label과의 거리를 함께 본다.
        if "soft_label" in df.columns and df["soft_label"].notna().any():
            soft = df["soft_label"].astype(float)
            prob = df["prob_hate"].astype(float)
            soft_acc = float(np.mean(1 - np.abs(soft - (prob > 0.5).astype(float))))
            # soft macro-F1 근사: soft label을 확률로 보고 hard 예측과의 기대 정밀도/재현율로 계산
            hard_pred = (prob > 0.5).astype(int)
            soft_tp = float(np.sum(soft * hard_pred))
            soft_fp = float(np.sum((1 - soft) * hard_pred))
            soft_fn = float(np.sum(soft * (1 - hard_pred)))
            soft_precision = soft_tp / (soft_tp + soft_fp) if (soft_tp + soft_fp) else 0.0
            soft_recall = soft_tp / (soft_tp + soft_fn) if (soft_tp + soft_fn) else 0.0
            soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) else 0.0
            mean_jsd = float(np.mean([jensen_shannon_divergence(p, s) for p, s in zip(prob, soft)]))
            print(f"soft accuracy: {soft_acc:.3f} | soft-F1(hate): {soft_f1:.3f} | mean JSD: {mean_jsd:.4f}")

        if args.error_analysis_n:
            # FN: true=hate but predicted non_hate, worst first (score most negative).
            # FP: true=non_hate but predicted hate, worst first (score most positive).
            fn = df[(y_true == 1) & (y_pred == 0)].sort_values("score")
            fp = df[(y_true == 0) & (y_pred == 1)].sort_values("score", ascending=False)
            print(f"\nFalse Negative {len(fn)}건 / False Positive {len(fp)}건")
            print(f"[FN 상위 {args.error_analysis_n}]\n", fn.head(args.error_analysis_n).to_string(index=False))
            print(f"[FP 상위 {args.error_analysis_n}]\n", fp.head(args.error_analysis_n).to_string(index=False))

            out_stem = os.path.splitext(args.out)[0]
            fn.to_csv(f"{out_stem}_fn.csv", index=False)
            fp.to_csv(f"{out_stem}_fp.csv", index=False)
            print(f"FN/FP saved to {out_stem}_fn.csv / {out_stem}_fp.csv")

    print(f"predictions saved to {args.out}")


if __name__ == "__main__":
    main()
