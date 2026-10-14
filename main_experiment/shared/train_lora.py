"""
LoRA fine-tune VARCO-VISION-2.0-1.7B (LlavaOnevisionForConditionalGeneration)
for hate / non_hate image+text classification, framed as generation of the
single token "혐오" or "비혐오".

이 파일은 박나현/조현서/정미정 세 사람이 공유하는 단일 train_lora.py다(shared/ 아래
1벌만 존재). 통합 매칭 설계(analysis/matched_design_analysis.py)에 따라 모델은
**딱 한 번만** 학습한다: data_prep.py --case-pairs로 만든 hate(박나현 케이스, 11) +
image_hate_text_safe(조현서 케이스, 10) + text_hate_image_safe(정미정 케이스, 01) +
non_hate를 모두 합친 하나의 train.jsonl로 학습해서, 세 조건이 서로 다른 모델이 아니라
동일한 checkpoint에서 나오게 한다. 이후 evaluate.py --eval-filter로 조건별 성능을
따로 뽑는다.

Designed to run on a Colab/Elice GPU (T4/A100). Requires (see requirements.txt):
    pip install -U "transformers==4.53.1" peft accelerate pillow pandas

Usage:
    python train_lora.py \
        --dataset-root ../image_dataset \
        --train-jsonl data/train.jsonl \
        --output-dir outputs/lora-adapter \
        --epochs 3 --batch-size 2 --grad-accum 16 --seed 42

Notes:
- By default only the language-model (Qwen3) attention/MLP projections get
  LoRA adapters; the SigLIP2 vision tower and multi_modal_projector stay
  frozen. Pass --tune-vision to also add LoRA adapters to the vision tower
  (more faithful to "train the image-part model", but heavier/slower and
  more prone to overfitting).
- FP shortcut-learning (e.g. flagging any mention of "다문화"/"성소수자"/
  "일베" regardless of context) is helped by focal loss, which down-weights
  easy/already-confident examples and up-weights confidently-wrong ones.
  Set --focal-gamma > 0 (e.g. 2.0) to enable; 0 (default) is plain CE loss.
  This works best combined with `data_prep.py --augment`, which adds the
  counterfactual (FP) and slang/변형어 (FN) examples this is meant to fix.
- Modality laziness(이미지만 보고 텍스트는 무시, 또는 그 반대)를 줄이기 위한 두 가지
  보완 장치:
    * --modality-dropout: train 표본마다 이 확률로 이미지를 회색으로 가려서 텍스트
      활용을 강제. 가린 표본의 정답은 CASE_MODALITY_TARGETS의 텍스트 단독 정답으로
      바뀐다(예: image_hate_text_safe -> 비혐오).
    * --aux-text-weight/--aux-image-weight: 매 스텝 "텍스트만 봤을 때 정답"/"이미지만
      봤을 때 정답"에 대한 auxiliary loss를 추가로 계산해 더한다. 케이스별 정답은
      CASE_MODALITY_TARGETS에 정의(모르는 케이스는 최종 정답으로 안전하게 폴백).

Two-stage training:
- Stage 1 always trains on the FULL train.jsonl (every text, every image,
  every case) -- this is the general hate/non_hate behavior across all three
  people's matched conditions.
- Stage 2 (optional, --case-epochs > 0) continues training the SAME model
  for a few more epochs on just the subset matching --case-label -- one of
  the three matched conditions (default "hate"). This reinforces that case
  on top of the general pass instead of replacing it, since stage 1's
  weights are what stage 2 continues from.
"""
import argparse
import datetime
import json
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from peft import LoraConfig, get_peft_model
from PIL import Image

import image_prep

MODEL_ID = "NCSOFT/VARCO-VISION-2.0-1.7B"

LANGUAGE_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VISION_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

# LLaVA-OneVision's "anyres" preprocessing splits large images into multiple
# high-res tiles (each ~576-729 vision tokens), which can blow past a T4's
# 16GB just from the lm_head (152k vocab) being applied at every position.
# Resizing to exactly the model's base tile size keeps it to a single tile.
IMAGE_SIZE = image_prep.IMAGE_SIZE

# --- Auxiliary modality loss (--aux-text-weight / --aux-image-weight) ---
# 각 케이스 label별로 "텍스트만 봤을 때 정답"/"이미지만 봤을 때 정답"이 이미 설계상
# 정해져 있다(학습으로 알아내는 게 아님). data_prep.py의 LABEL_TO_TARGET과 짝을 이루는
# 테이블: label -> (텍스트 단독 정답, 이미지 단독 정답). 여기 없는 label은 아래
# get_modality_targets()에서 안전하게 "본 라벨의 target과 동일"으로 폴백한다 -- 즉 aux
# loss가 최종 정답과 다른 방향으로 학습을 방해하는 일은 없다(모르면 그냥 본 loss와
# 동일하게 취급).
CASE_MODALITY_TARGETS = {
    "hate": ("혐오", "혐오"),                        # 박나현 케이스: 텍스트도 유해, 이미지도 유해
    "non_hate": ("비혐오", "비혐오"),                  # 둘 다 무해
    "image_hate_text_safe": ("비혐오", "혐오"),        # 조현서 케이스: 텍스트=무해, 이미지=유해
    "text_hate_image_safe": ("혐오", "비혐오"),        # 정미정 케이스: 텍스트=유해, 이미지=무해
}

IMAGE_ONLY_PROMPT = (
    "다음 이미지를 보고 혐오 표현(혐오, 차별, 비하 등)을 담고 있는지 판단하세요. "
    "반드시 '혐오' 또는 '비혐오' 중 하나의 단어로만 답하세요."
)


def get_modality_targets(label, fallback_target):
    """(텍스트 단독 정답, 이미지 단독 정답)을 반환. 모르는 label이면 최종 정답으로 폴백."""
    return CASE_MODALITY_TARGETS.get(label, (fallback_target, fallback_target))


def split_train_val(records, val_frac, seed):
    """train.jsonl -> (train_records, val_records), grouped by match_cluster_id
    (data_prep.py가 image_group_id/text_group_id 공유 관계로 계산한 연결 요소 -- 같은
    이미지 또는 같은 텍스트를 공유하는 표본은 같은 split에 배치돼야 실제 내용 누수가
    없다) and stratified by label. match_cluster_id가 없는 구버전 jsonl에서는 id를
    그대로 클러스터로 취급한다.

    주의: data_prep.py --case-pairs는 hate/non_hate 표본의 이미지·텍스트를 거의
    전부 재사용해 조현서/정미정 케이스를 만들기 때문에, 이 연결 요소 중 일부가 수백
    개 표본을 묶는 "거대 클러스터"가 되는 게 정상이다(버그가 아니라 모든 항목을
    한 번씩 재사용하는 크로스 페어링 설계의 구조적 결과). 거대 클러스터를 억지로
    쪼개 train/val에 나누면 그게 진짜 누수이므로, val은 "통째로 넣어도 목표
    val_frac을 넘지 않는" 작은 클러스터들로만 채우고(작은 것부터 그리디로, 넘치는
    클러스터는 건너뜀), 그래도 목표에 못 미치면 val이 작아진 채로 두고 경고를 남긴다
    -- 거대 클러스터 일부를 쪼개서 목표치를 억지로 맞추는 것보다 훨씬 안전하다.

    결정성(reproducibility) 주의: Python의 `set`은 문자열 원소의 경우 반복 순서가
    PYTHONHASHSEED(프로세스마다 랜덤)에 따라 달라진다. 라벨 순회 순서나 동점 라벨
    선택에 `set(...)`을 그대로 쓰면, 여러 라벨이 하나의 rng를 순서대로 소비하면서
    "같은 seed인데도 실행마다 val 비율이 달라지는" 문제가 생긴다(실측: 0.28~0.69로
    출렁임). 그래서 (1) 라벨은 `sorted(set(...))`로 고정 순서로 순회하고, (2) 라벨마다
    독립적으로 시드된 rng(`random.Random(f"{seed}:{label}")` -- 문자열 시드는 내부적으로
    sha512 해시를 쓰므로 PYTHONHASHSEED와 무관하게 프로세스 간에도 결정적이다)를 써서
    한 라벨의 셔플이 다른 라벨의 셔플에 전혀 영향을 못 주게 한다."""
    clusters = {}
    for r in records:
        cid = r.get("match_cluster_id", r["id"])
        clusters.setdefault(cid, []).append(r)

    # 클러스터 단위 라벨(클러스터 내 다수 라벨)로 층화. 동점이면 sorted로 고정 선택.
    cluster_label = {}
    for cid, recs in clusters.items():
        labels = [r["label"] for r in recs]
        counts = Counter(labels)
        best = max(counts.values())
        cluster_label[cid] = sorted(l for l, c in counts.items() if c == best)[0]

    val_cids = set()
    for label in sorted(set(cluster_label.values())):
        label_rng = random.Random(f"{seed}:{label}")
        cids = [c for c, l in cluster_label.items() if l == label]
        label_rng.shuffle(cids)
        # 작은 클러스터부터 채워야(그리디) 큰 덩어리를 어쩔 수 없이 통째로 val에 넣어
        # val_frac을 크게 초과시키는 상황을 피할 수 있다.
        cids.sort(key=lambda c: len(clusters[c]))
        total_n = sum(len(clusters[c]) for c in cids)
        target_n = round(total_n * val_frac)
        picked_n = 0
        for c in cids:
            size = len(clusters[c])
            # 넘치는 클러스터는 억지로 넣지 않고 건너뛴다(오름차순 정렬이라 이후
            # 클러스터는 크기가 같거나 더 커서, 하나가 못 들어가면 나머지도 다
            # 못 들어간다 -- break로 종료).
            if picked_n + size > target_n:
                break
            val_cids.add(c)
            picked_n += size
        if target_n > 0 and picked_n == 0:
            smallest = min(len(clusters[c]) for c in cids)
            print(f"[validation 분리][경고] label={label}: 목표({target_n}개)를 넘지 않는 "
                  f"클러스터가 하나도 없어(가장 작은 클러스터도 {smallest}개) val에 이 "
                  "라벨을 못 넣었습니다 -- 거대 클러스터를 쪼개는 대신 전부 train에 둡니다.")

    train_records, val_records = [], []
    for cid, recs in clusters.items():
        (val_records if cid in val_cids else train_records).extend(recs)

    actual_frac = len(val_records) / len(records) if records else 0.0
    if val_records and abs(actual_frac - val_frac) > 0.05:
        print(f"[validation 분리][참고] 목표 val_frac={val_frac:.2f}였지만 누수 없는 클러스터 "
              f"경계로만 채운 결과 실제 val 비율은 {actual_frac:.2f}입니다(거대 클러스터를 "
              "쪼개는 대신 작은 클러스터로만 채웠기 때문 -- data_prep.py의 --case-pairs "
              "크로스 페어링 특성상 정상입니다).")
    return train_records, val_records


class HateVLMDataset(Dataset):
    def __init__(self, jsonl_path_or_records, dataset_root: str, label_filter: str = None, split: str = "train",
                 modality_dropout: float = 0.0):
        if isinstance(jsonl_path_or_records, str):
            records = [json.loads(l) for l in open(jsonl_path_or_records, "r", encoding="utf-8")]
        else:
            records = list(jsonl_path_or_records)
        if label_filter is not None:
            records = [r for r in records if r["label"] == label_filter]

        # 손상 파일은 여기서 미리 걸러 로그로 남긴다 (연구 설계서: "손상 파일은
        # 예외 처리하고 해당 sample_id와 사유를 기록").
        good_records = []
        for r in records:
            full_path = os.path.join(dataset_root, r["image_path"])
            if image_prep.load_image_for_model(full_path, sample_id=r["id"]) is not None:
                good_records.append(r)
        self.records = good_records
        self.dataset_root = dataset_root
        self.split = split  # "train"이면 augmentation transform, 그 외엔 결정론적 transform
        # modality dropout: train에서만, 표본마다 이 확률로 이미지를 회색으로 가려서
        # "이미지만 봐도 되는" 지름길을 못 쓰게 하고 텍스트도 강제로 활용하게 만든다.
        # (validation/test에는 절대 적용하지 않는다 -- 평가는 항상 실제 입력 그대로 해야 함)
        self.modality_dropout = modality_dropout if split == "train" else 0.0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        target = r["target"]
        if self.modality_dropout > 0 and random.random() < self.modality_dropout:
            # 텍스트만 남기고 이미지는 정보 없는 회색으로 대체 (evaluate.py --blank-image와 동일한 회색값)
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
            # 이미지를 가렸으면 정답도 "텍스트만 봤을 때 정답"으로 바꿔야 한다. 이전 버전은
            # 최종 정답(OR 규칙)을 그대로 둬서, image_hate_text_safe(유해 이미지+무해 텍스트)
            # 표본이 "회색 이미지 + 무해 텍스트 -> 혐오"로 학습됐다 -- 즉 "회색 이미지면
            # 혐오"라는 지름길을 가르쳤고, 실제로 seed1/3에서 이 조건의 text-only 채점이
            # 100% 혐오로 나왔다.
            target, _ = get_modality_targets(r["label"], r["target"])
        else:
            image = image_prep.prepare_image(
                os.path.join(self.dataset_root, r["image_path"]), self.split, sample_id=r["id"],
            )
        return {"id": r["id"], "label": r["label"], "image": image, "prompt": r["prompt"], "target": target}


def build_collate_fn(processor, build_aux=False):
    """build_aux=True이면 배치에 'aux_text'/'aux_image' 서브배치를 추가로 만든다
    (--aux-text-weight/--aux-image-weight로 auxiliary modality loss를 쓸 때만 필요.
    안 쓰면 매 스텝 forward pass가 3배로 늘어나는 걸 피하려고 기본은 False)."""

    def _encode_one(image, prompt, target):
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        full_text = prompt_text + target

        prompt_enc = processor(text=prompt_text, images=image, return_tensors="pt")
        full_enc = processor(text=full_text, images=image, return_tensors="pt")

        input_ids = full_enc["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100  # only supervise the target token(s)
        return input_ids, labels, full_enc["attention_mask"][0], full_enc["pixel_values"][0], full_enc.get("image_sizes")

    def _pack(input_ids_list, labels_list, attn_mask_list, pixel_values_list, image_sizes_list):
        max_len = max(x.shape[0] for x in input_ids_list)
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

        def pad(seqs, pad_value):
            out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
            for i, s in enumerate(seqs):
                out[i, : s.shape[0]] = s
            return out

        out = {
            "input_ids": pad(input_ids_list, pad_id),
            "attention_mask": pad(attn_mask_list, 0),
            "labels": pad(labels_list, -100),
            "pixel_values": torch.stack(pixel_values_list),
        }
        if image_sizes_list and image_sizes_list[0] is not None:
            out["image_sizes"] = torch.stack([s[0] for s in image_sizes_list])
        return out

    def collate(batch):
        input_ids_list, labels_list, pixel_values_list, image_sizes_list = [], [], [], []
        attn_mask_list = []
        aux_text_lists = {"input_ids": [], "labels": [], "attn": [], "pix": [], "sizes": []}
        aux_image_lists = {"input_ids": [], "labels": [], "attn": [], "pix": [], "sizes": []}

        for ex in batch:
            ids, labs, attn, pix, sizes = _encode_one(ex["image"], ex["prompt"], ex["target"])
            input_ids_list.append(ids); labels_list.append(labs)
            attn_mask_list.append(attn); pixel_values_list.append(pix)
            if sizes is not None:
                image_sizes_list.append(sizes)

            if build_aux:
                text_target, image_target = get_modality_targets(ex["label"], ex["target"])
                # 텍스트-only 서브배치: 이미지를 회색으로 가리고, 정답은 "텍스트만 봤을 때" 정답
                blank_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
                t_ids, t_labs, t_attn, t_pix, t_sizes = _encode_one(blank_image, ex["prompt"], text_target)
                aux_text_lists["input_ids"].append(t_ids); aux_text_lists["labels"].append(t_labs)
                aux_text_lists["attn"].append(t_attn); aux_text_lists["pix"].append(t_pix)
                if t_sizes is not None:
                    aux_text_lists["sizes"].append(t_sizes)

                # 이미지-only 서브배치: 실제 이미지 + 텍스트 없는 프롬프트, 정답은 "이미지만 봤을 때" 정답
                i_ids, i_labs, i_attn, i_pix, i_sizes = _encode_one(ex["image"], IMAGE_ONLY_PROMPT, image_target)
                aux_image_lists["input_ids"].append(i_ids); aux_image_lists["labels"].append(i_labs)
                aux_image_lists["attn"].append(i_attn); aux_image_lists["pix"].append(i_pix)
                if i_sizes is not None:
                    aux_image_lists["sizes"].append(i_sizes)

        batch_out = {
            "ids": [ex["id"] for ex in batch],  # 텐서 아님 -- hard-example 가중치 조회용, device로 안 옮김
            "targets": [ex["target"] for ex in batch],  # 텐서 아님 -- --class-balance 가중치 조회용
            **_pack(input_ids_list, labels_list, attn_mask_list, pixel_values_list, image_sizes_list),
        }
        if build_aux:
            batch_out["aux_text"] = _pack(
                aux_text_lists["input_ids"], aux_text_lists["labels"], aux_text_lists["attn"],
                aux_text_lists["pix"], aux_text_lists["sizes"],
            )
            batch_out["aux_image"] = _pack(
                aux_image_lists["input_ids"], aux_image_lists["labels"], aux_image_lists["attn"],
                aux_image_lists["pix"], aux_image_lists["sizes"],
            )
        return batch_out

    return collate


def focal_ce_loss(logits, labels, gamma=0.0, alpha=1.0, label_smoothing=0.0, sample_weights=None):
    """Causal-LM cross-entropy over the (already -100-masked) target tokens,
    reweighted by the focal loss factor (1-p_t)^gamma. gamma=0 reduces
    exactly to plain mean CE. label_smoothing > 0 caps how close p_t can get
    to 1.0, which keeps the model from collapsing into extreme-confidence
    single-answer predictions (e.g. logp_hate -> 0 regardless of input).

    sample_weights: (batch,) 텐서. 주어지면 표본별 토큰 loss에 곱한다(--class-balance).
    분모는 가중치 합이 아니라 토큰 수 그대로 둔다 -- 가중치 합으로 나누면 한 클래스만
    든 배치(batch_size=2에서 흔함)에서 가중치가 약분돼 균형 효과가 사라진다. 클래스
    가중치는 데이터 전체 평균이 1이 되게 만들어 loss 스케일은 유지된다."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    mask = (shift_labels.view(-1) != -100)
    # ignore_index only guarantees the *nll* term is exact at masked positions;
    # with label_smoothing > 0, PyTorch's smoothing term is computed from
    # log_softmax over ALL classes regardless of ignore_index, so a masked
    # position whose (arbitrary, un-supervised) logits happen to overflow can
    # produce inf/nan there. `ce * mask.float()` does NOT save us -- 0 * nan
    # == nan in IEEE float, so it would silently poison the whole batch's
    # loss. torch.where selects elementwise instead of multiplying, so a
    # nan/inf at a masked position is simply discarded, not combined.
    ce = torch.where(mask, ce, torch.zeros_like(ce))
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = alpha * (1 - pt) ** gamma * ce
        ce = torch.where(mask, ce, torch.zeros_like(ce))
    if sample_weights is not None:
        w = sample_weights.to(device=ce.device, dtype=ce.dtype).unsqueeze(1).expand(shift_labels.shape).reshape(-1)
        ce = torch.where(mask, ce * w, torch.zeros_like(ce))
    denom = mask.float().sum().clamp(min=1.0)
    return ce.sum() / denom


def build_optimizer_and_scheduler(model, loader, lr, epochs, grad_accum, vision_lr_scale=1.0):
    """vision_lr_scale != 1.0이면 "vision_tower"가 이름에 들어간 학습 파라미터(--tune-vision
    LoRA)를 별도 param group으로 분리해 lr*vision_lr_scale을 적용한다. 비전 타워가 이미
    강한 신호를 갖고 있어 과적합되기 쉬우므로, 1.0 미만으로 낮춰 언어모델 쪽보다 천천히
    학습시키는 용도."""
    if vision_lr_scale != 1.0:
        vision_params = [p for n, p in model.named_parameters() if p.requires_grad and "vision_tower" in n]
        other_params = [p for n, p in model.named_parameters() if p.requires_grad and "vision_tower" not in n]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": lr},
            {"params": vision_params, "lr": lr * vision_lr_scale},
        ])
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = (len(loader) // grad_accum) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    return optimizer, scheduler


@torch.no_grad()
def eval_val_loss(model, val_loader, device):
    """early stopping/checkpoint 선택용 validation loss (plain CE, focal 가중치 없이).
    Hate-CLIPper 관례대로 validation은 오직 모델/하이퍼파라미터 선택에만 쓴다."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in val_loader:
        batch = dict(batch)
        batch.pop("ids", None)
        batch.pop("targets", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        if device == "cuda":
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        labels = batch.pop("labels")
        outputs = model(**batch, use_cache=False)
        loss = focal_ce_loss(outputs.logits, labels, gamma=0.0)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_loop(model, loader, optimizer, scheduler, epochs, grad_accum, focal_gamma, focal_alpha, device,
               phase_name, step_offset=0, label_smoothing=0.0, val_loader=None, processor=None,
               early_stopping_patience=0, best_checkpoint_dir=None, hard_example_weights=None,
               aux_text_weight=0.0, aux_image_weight=0.0, class_weights=None):
    """Runs `epochs` passes over `loader`, updating `model` in place. Shared
    by both stage 1 (full data) and stage 2 (case-only) -- stage 2 just calls
    this again with a different loader/optimizer/scheduler on the same model,
    so it continues from stage 1's weights instead of starting over.

    val_loader가 주어지면 매 epoch 끝에 validation loss를 측정하고, 개선될 때마다
    best_checkpoint_dir에 어댑터를 저장한다(체크포인트 저장 확인 항목). 개선이
    early_stopping_patience epoch 연속 없으면 학습을 조기 종료한다(0이면 비활성화).

    hard_example_weights: {sample_id: weight} 딕셔너리. 주어지면 그 표본이 포함된 스텝의
    loss에 가중치를 곱한다 (이전 라운드에서 FN이었던 hard positive를 더 세게 학습시키는 용도).
    --batch-size 1을 전제로, 배치 안 표본들 가중치의 평균을 씀(batch_size>1이면 근사치).
    딕셔너리에 없는 id는 가중치 1.0(기본)으로 취급한다.

    aux_text_weight/aux_image_weight: 0보다 크면, 본 배치(text+image -> y_MM) 외에
    collate_fn이 build_aux=True로 만들어준 batch["aux_text"](이미지 회색+텍스트만으로
    y_text 예측), batch["aux_image"](텍스트 없이 이미지만으로 y_image 예측)에 대해서도
    각각 forward pass를 추가로 돌려서 L = L_MM + aux_text_weight*L_text + aux_image_weight*L_image
    로 합친다. loader의 collate_fn이 build_aux=True로 만들어진 경우에만 batch에 이 키들이
    있으므로, 가중치가 0보다 크면 반드시 그런 collate_fn을 쓰는 loader를 넘겨야 한다.

    class_weights: {"혐오": w, "비혐오": w} 딕셔너리(--class-balance). 주어지면 본 배치
    loss에 표본별로(그 표본의 실제 학습 정답 기준 -- modality dropout으로 바뀐 정답 포함)
    가중치를 곱한다. aux loss에는 적용하지 않는다."""
    model.train()
    step = step_offset
    best_val_loss = float("inf")
    epochs_no_improve = 0
    hard_example_weights = hard_example_weights or {}
    use_aux = aux_text_weight > 0 or aux_image_weight > 0

    def _forward_loss(sub_batch):
        sub_batch = dict(sub_batch)
        sub_batch = {k: v.to(device) for k, v in sub_batch.items()}
        if device == "cuda":
            sub_batch["pixel_values"] = sub_batch["pixel_values"].to(torch.bfloat16)
        sub_labels = sub_batch.pop("labels")
        sub_outputs = model(**sub_batch, use_cache=False)
        return focal_ce_loss(sub_outputs.logits, sub_labels, gamma=focal_gamma, alpha=focal_alpha,
                              label_smoothing=label_smoothing)

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            batch = dict(batch)
            ids = batch.pop("ids", None)
            targets = batch.pop("targets", None)
            aux_text_batch = batch.pop("aux_text", None)
            aux_image_batch = batch.pop("aux_image", None)

            batch = {k: v.to(device) for k, v in batch.items()}
            if device == "cuda":
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
            labels = batch.pop("labels")
            outputs = model(**batch, use_cache=False)
            sample_weights = None
            if class_weights and targets:
                sample_weights = torch.tensor([class_weights.get(t, 1.0) for t in targets])
            loss = focal_ce_loss(outputs.logits, labels, gamma=focal_gamma, alpha=focal_alpha,
                                 label_smoothing=label_smoothing, sample_weights=sample_weights)

            if use_aux and aux_text_weight > 0 and aux_text_batch is not None:
                loss = loss + aux_text_weight * _forward_loss(aux_text_batch)
            if use_aux and aux_image_weight > 0 and aux_image_batch is not None:
                loss = loss + aux_image_weight * _forward_loss(aux_image_batch)

            if ids and hard_example_weights:
                w = sum(hard_example_weights.get(sid, 1.0) for sid in ids) / len(ids)
                loss = loss * w

            loss = loss / grad_accum
            loss.backward()
            running_loss += loss.item()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"[{phase_name}] epoch {epoch} step {step} loss {running_loss:.4f}")
                running_loss = 0.0

        print(f"=== [{phase_name}] epoch {epoch} done ===")

        if val_loader is not None:
            val_loss = eval_val_loss(model, val_loader, device)
            print(f"[{phase_name}] epoch {epoch} val_loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if best_checkpoint_dir is not None:
                    os.makedirs(best_checkpoint_dir, exist_ok=True)
                    model.save_pretrained(best_checkpoint_dir)
                    if processor is not None:
                        processor.save_pretrained(best_checkpoint_dir)
                    print(f"[{phase_name}] val_loss 개선 -> best checkpoint 저장: {best_checkpoint_dir}")
            else:
                epochs_no_improve += 1
                if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                    print(f"[{phase_name}] val_loss가 {epochs_no_improve} epoch 연속 개선되지 않아 "
                          f"early stopping (best_val_loss={best_val_loss:.4f}).")
                    break
    return step, best_val_loss


@torch.no_grad()
def _sequence_logprob(model, processor, image, prompt_text, target_text, device):
    """evaluate.py의 sequence_logprob과 동일한 채점 방식(teacher-forced log-likelihood)
    -- threshold를 validation에서 고르려면 test 채점과 완전히 같은 방식으로 val을 채점해야
    임계값이 실제로 맞아떨어진다."""
    full_text = prompt_text + target_text
    prompt_enc = processor(text=prompt_text, images=image, return_tensors="pt")
    full_enc = processor(text=full_text, images=image, return_tensors="pt")
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    pixel_values = full_enc["pixel_values"].to(device)
    if device == "cuda":
        pixel_values = pixel_values.to(torch.bfloat16)
    kwargs = {"pixel_values": pixel_values}
    if "image_sizes" in full_enc:
        kwargs["image_sizes"] = full_enc["image_sizes"].to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    n_target_tokens = (labels != -100).sum().item()
    return -outputs.loss.item() * n_target_tokens


def select_threshold_on_val(model, processor, val_records, dataset_root, device,
                             strategy="f1", min_precision=0.99, max_fpr=0.01):
    """validation에서 hate-score(logp_hate - logp_nonhate) 임계값을 고른다.

    strategy="f1"(기본값): F1이 제일 높은 지점을 고름.
    strategy="balanced_accuracy": (recall + 1 - FPR)/2가 제일 높은 지점을 고름(클래스 불균형에 강건).
    strategy="precision_constrained": Precision >= min_precision을 만족하는 후보들 중에서
        Recall이 가장 높은 지점을 고름. F1 기준으로 골랐을 때 threshold가 지나치게
        높아져서(=모델이 과하게 조심스러워져서) Recall이 떨어지는 문제를 완화하는 용도.
        만족하는 후보가 하나도 없으면 f1 기준으로 자동 폴백한다.

    두 전략 모두 test.jsonl은 절대 건드리지 않고 validation에서만 임계값을 고른 뒤(설계서
    6장: "validation에서 임계값을 선택하고 test에는 고정된 임계값을 그대로 적용"), 그
    임계값을 test에 고정 적용한다. model.eval() 상태에서 호출해야 한다.
    """
    model.eval()
    scores, labels = [], []
    for r in val_records:
        image = image_prep.prepare_image(os.path.join(dataset_root, r["image_path"]), "val", sample_id=r["id"])
        if image is None:
            continue
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": r["prompt"]},
        ]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        logp_hate = _sequence_logprob(model, processor, image, prompt_text, "혐오", device)
        logp_nonhate = _sequence_logprob(model, processor, image, prompt_text, "비혐오", device)
        scores.append(logp_hate - logp_nonhate)
        # true_target(실제 정답 토큰) 기준으로 유해 여부 판정 -- label이 literally "hate"가
        # 아닌 케이스(image_hate_text_safe/text_hate_image_safe)도 정확히 반영되어야 함.
        labels.append(1 if r["target"] == "혐오" else 0)
    model.train()

    if not scores or len(set(labels)) < 2:
        print("[threshold 선택] validation 표본이 부족하거나 단일 클래스라 기본 임계값 0.0을 씁니다.")
        return 0.0, None

    candidates = sorted(set(scores))

    def metrics_at(t):
        preds = [1 if s > t else 0 for s in scores]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, fpr, f1

    if strategy == "precision_constrained":
        feasible = []
        for t in candidates:
            precision, recall, fpr, f1 = metrics_at(t)
            if precision >= min_precision:
                feasible.append((t, precision, recall, fpr, f1))
        if feasible:
            # recall이 동점인 후보가 여럿이면(흔함 -- 작은 val set에서 recall=1.0을
            # 만족하는 threshold 구간이 넓을 수 있음) 그중 **가장 높은(보수적인)**
            # threshold를 고른다. 이전 버전은 `-x[0]`(threshold 오름차순 -> 가장 낮은/
            # 관대한 threshold 선택)이라 "precision을 지키면서 recall 최대화"라는
            # 의도와 반대로 가장 관대한 지점을 골라 FPR이 오히려 f1 전략보다 나빠지는
            # 문제가 있었다(실측: min_precision=0.9로도 non_hate FPR이 33%->48%로 악화).
            best_t, best_p, best_r, best_fpr, best_f1 = max(feasible, key=lambda x: (x[2], x[0]))
            print(f"[threshold 선택: precision_constrained] validation n={len(scores)} "
                  f"best_threshold={best_t:.4f} precision={best_p:.3f} recall={best_r:.3f} "
                  f"fpr={best_fpr:.3f} (min_precision={min_precision})")
            return best_t, best_f1
        print(f"[threshold 선택: precision_constrained] precision>={min_precision}을 만족하는 "
              f"threshold가 없어 f1 기준으로 폴백합니다.")

    if strategy == "balanced_accuracy":
        # (recall + (1 - FPR)) / 2 최대화. --case-pairs 데이터는 val도 정답의 약 3/4이 혐오라,
        # F1(양성 클래스 기준) 최댓값은 임계값을 혐오 쪽으로 크게 끌어내려 non_hate FPR을
        # 키운다(seed1 실측: f1 선택 -1.88 -> test FPR 34.7%). balanced accuracy는 두 클래스를
        # 같은 비중으로 보므로 클래스 비율에 영향을 받지 않는다. 동점이면 높은(보수적인)
        # threshold를 고른다.
        best = max(
            ((t, *metrics_at(t)) for t in candidates),
            key=lambda x: ((x[2] + (1 - x[3])) / 2, x[0]),
        )
        best_t, best_p, best_r, best_fpr, best_f1 = best
        print(f"[threshold 선택: balanced_accuracy] validation n={len(scores)} "
              f"best_threshold={best_t:.4f} bacc={(best_r + 1 - best_fpr) / 2:.3f} "
              f"recall={best_r:.3f} fpr={best_fpr:.3f} f1={best_f1:.3f}")
        return best_t, best_f1

    best_t, best_f1 = 0.0, -1.0
    for t in candidates:
        _, _, _, f1 = metrics_at(t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"[threshold 선택: f1] validation n={len(scores)} best_threshold={best_t:.4f} val_F1={best_f1:.3f}")
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--train-jsonl", default="data/train.jsonl")
    ap.add_argument("--output-dir", default="outputs/lora-adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--tune-vision", action="store_true", help="Also add LoRA to the SigLIP2 vision tower")
    ap.add_argument(
        "--image-dropout-prob", type=float, default=0.0,
        help="--modality-dropout의 별칭(하위호환용). --modality-dropout과 동시에 주면 "
        "--modality-dropout이 우선합니다.",
    )
    ap.add_argument(
        "--vision-lr-scale", type=float, default=1.0,
        help="비전 타워(--tune-vision) LoRA 파라미터에 곱할 학습률 배수. 1.0(기본)이면 "
        "언어모델 쪽과 동일한 lr. 비전 타워가 상대적으로 이미 강한 신호(AUROC 0.99대)를 "
        "가지고 있어 과적합되기 쉬우므로, 0.5 등으로 낮춰서 언어모델 쪽이 더 빨리 배우게 "
        "할 수 있습니다.",
    )
    ap.add_argument(
        "--no-grad-checkpointing", dest="grad_checkpointing", action="store_false",
        help="Disable gradient checkpointing (on by default; turning it off is faster but uses much more GPU memory).",
    )
    ap.add_argument(
        "--focal-gamma", type=float, default=0.0,
        help="Focal loss gamma for FP shortcut-learning (e.g. 2.0). 0 = plain CE (default).",
    )
    ap.add_argument("--focal-alpha", type=float, default=1.0, help="Focal loss alpha weight")
    ap.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="0~1. 타깃 토큰 확률이 1.0에 극단적으로 수렴하는 것을 막아 과확신(예: logp_hate가 "
        "거의 0으로 붕괴)을 완화합니다. 0(기본값)이면 기존과 동일. 0.05~0.1 정도 권장.",
    )
    ap.add_argument(
        "--modality-dropout", type=float, default=0.0,
        help="0~1. stage 1 train 표본마다 이 확률로 이미지를 정보 없는 회색으로 가리고 텍스트만으로 "
        "학습시킵니다. 모델이 '이미지만 봐도 되네'라는 지름길을 못 쓰게 강제해서, 텍스트 신호가 "
        "있는데도 안 쓰는 modality laziness/편향을 줄이는 용도입니다. validation/test에는 절대 "
        "적용 안 됨(평가는 항상 실제 입력 그대로). 0(기본값)이면 기존과 동일. 0.15~0.3 정도부터 "
        "시도해보고, modality_bias_report.py로 자연 케이스 텍스트-only AUROC이 오르는지 확인하세요.",
    )
    ap.add_argument(
        "--threshold-strategy", choices=["f1", "precision_constrained", "balanced_accuracy"], default="f1",
        help="validation 임계값 선택 기준. 'f1'(기본)은 F1 최댓값 지점. "
        "'balanced_accuracy'는 (recall + 1 - FPR)/2 최댓값 지점으로, val이 혐오 쪽으로 치우친 "
        "--case-pairs 데이터에서 F1 기준이 임계값을 과하게 낮추는 문제를 피합니다. "
        "'precision_constrained'는 --min-precision 조건을 만족하는 후보 중 Recall이 제일 "
        "높은 지점을 고릅니다 -- F1 기준으로 임계값이 과하게 높아져서(모델이 지나치게 "
        "조심스러워져서) Recall이 떨어지는 문제를 완화하는 용도입니다.",
    )
    ap.add_argument(
        "--min-precision", type=float, default=0.99,
        help="--threshold-strategy precision_constrained일 때 쓰는 최소 precision 기준.",
    )
    ap.add_argument(
        "--max-fpr", type=float, default=0.01,
        help="(참고용, 단독 판정 기준으로는 안 씀) --threshold-strategy precision_constrained의 "
        "리포트에 함께 표시할 최대 FPR 기준.",
    )
    ap.add_argument(
        "--hard-example-ids", default=None,
        help="이전 라운드에서 FN(놓친 hate)이었던 train-set 표본 id 목록 파일(한 줄에 하나씩). "
        "주어지면 그 표본이 포함된 학습 스텝의 loss에 --hard-example-weight를 곱해 더 세게 "
        "학습시킨다. 반드시 TRAIN set 기준 FN이어야 함(test set FN을 쓰면 데이터 누수). "
        "만드는 법: evaluate.py --test-jsonl data/train.jsonl로 train set을 채점해서 FN id를 뽑는다.",
    )
    ap.add_argument(
        "--hard-example-weight", type=float, default=1.75,
        help="--hard-example-ids로 지정된 표본의 loss 가중치. 1.5~2.0 권장(기본 1.75). "
        "너무 크게 주면 그 표본들에 과적합될 수 있음.",
    )
    ap.add_argument(
        "--aux-text-weight", type=float, default=0.0,
        help="0보다 크면 stage 1에서 매 스텝마다 '이미지 회색 처리 + 텍스트만으로 y_text 예측' "
        "auxiliary loss를 추가로 계산해서 L = L_MM + aux_text_weight*L_text (+ aux_image_weight*L_image)로 "
        "합칩니다. 케이스별 y_text 정답은 CASE_MODALITY_TARGETS에 정의됨(모르는 케이스는 최종 정답으로 "
        "폴백, 즉 안전한 기본값). 이미지 신호를 죽이지 않으면서 텍스트도 제대로 배우게 하는 용도. "
        "0(기본값)이면 기존과 동일(aux forward pass 자체를 안 돔). 0.3 정도부터 시도해보세요. "
        "매 스텝 forward pass가 최대 3배로 늘어나 학습 시간도 그만큼 늘어납니다.",
    )
    ap.add_argument(
        "--aux-image-weight", type=float, default=0.0,
        help="0보다 크면 '텍스트 없이 이미지만으로 y_image 예측' auxiliary loss를 추가합니다. "
        "--aux-text-weight와 같이 쓰는 걸 전제로 함(이미지는 이미 신호가 강해 보통 텍스트보다 "
        "낮은 가중치를 줌, 예: 0.1). 0(기본값)이면 비활성화.",
    )
    ap.add_argument(
        "--class-balance", action="store_true",
        help="stage 1 loss에 정답 클래스(혐오/비혐오) 역빈도 가중치를 곱합니다. --case-pairs 데이터는 "
        "정답이 '혐오'인 표본(hate+image_hate_text_safe+text_hate_image_safe)이 non_hate의 약 3배라, "
        "가중치 없이 학습하면 모델이 혐오 쪽으로 기울어 non_hate FPR이 높아집니다(seed1~4 실측 "
        "FPR 11~40%%). 가중치는 N/(2*N_class)로, 데이터 전체 평균이 1이 되게 맞춥니다.",
    )
    ap.add_argument(
        "--case-label",
        choices=["hate", "non_hate", "image_hate_text_safe", "text_hate_image_safe"],
        default="hate",
        help="담당 케이스 라벨 -- stage 2(케이스 전용 추가 학습)에서 --train-jsonl 중 이 label만 "
        "다시 도는 서브셋을 고릅니다. 'hate'=박나현 케이스, 'image_hate_text_safe'=조현서 케이스 "
        "(data_prep.py --case-pairs로 생성), 'text_hate_image_safe'=정미정 케이스(동일). "
        "각각 evaluate.py --eval-filter <값>과 동일 기준. --case-epochs=0이면 안 쓰임.",
    )
    ap.add_argument(
        "--case-epochs", type=int, default=0,
        help="stage 1(전체 데이터, --epochs만큼) 학습이 끝난 뒤, --case-label 서브셋에 이어서 "
        "추가로 도는 epoch 수. 0(기본값)이면 stage 2를 건너뛰고 기존과 동일하게 전체 데이터만 학습.",
    )
    ap.add_argument(
        "--case-lr", type=float, default=None,
        help="stage 2 학습률. 미지정 시 --lr의 절반 -- 이미 전체 데이터로 한 번 수렴한 가중치를 "
        "작은 서브셋으로 더 미세조정하는 단계라, --lr을 그대로 쓰면 과적합/재앙적 망각 위험이 큽니다.",
    )
    ap.add_argument(
        "--case-balance", action="store_true",
        help="stage 2가 --case-label 단일 클래스만으로 도는 것을 막기 위해, 반대 라벨(비혐오 "
        "케이스가 아니면 non_hate, 그 외는 서로) 예시를 같은 개수만큼 섞습니다. 단일 클래스로만 "
        "학습하면 loss를 0으로 만드는 가장 쉬운 길이 '입력과 무관하게 항상 case-label로 답하기'가 "
        "되어 모델이 그 라벨만 찍는 상수 함수로 붕괴(collapse)할 수 있습니다 -- 반대 라벨 예시가 "
        "이 지름길을 막는 반례 역할을 합니다.",
    )
    ap.add_argument(
        "--val-frac", type=float, default=0.15,
        help="train.jsonl에서 match_cluster_id(누수 방지 클러스터) 단위·라벨 층화로 "
        "떼어낼 validation 비율(거대 클러스터 특성상 실제 비율은 목표보다 작아질 수 "
        "있음 -- split_train_val 참고). 0이면 validation/early stopping/threshold "
        "선택을 모두 건너뛰고 기존 방식대로 전체 데이터를 stage 1에 그대로 씁니다.",
    )
    ap.add_argument(
        "--early-stopping-patience", type=int, default=2,
        help="stage 1에서 validation loss가 이 epoch 수만큼 연속 개선되지 않으면 조기 종료. "
        "0이면 비활성화(모든 epoch을 끝까지 돎). --val-frac=0이면 무시됩니다.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    modality_dropout = args.modality_dropout if args.modality_dropout > 0 else args.image_dropout_prob

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # fp16은 VARCO-VISION 순전파 중 오버플로우로 NaN이 나는 경우가 있어 bf16으로 변경.
    # A100은 bf16을 하드웨어 가속으로 지원해서 fp32만큼 안전하면서 속도 손해가 거의 없다.
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # use_fast=True: 이 체크포인트는 기본으로 순수 Python "slow" 이미지 프로세서를 쓰는데,
    # anyres 타일링(이미지 여러 장으로 쪼개기)이 표본마다 CPU에서 반복되면서 그게 병목이
    # 되어 학습이 극단적으로 느려지는 걸 실측으로 확인했다(스텝 하나 진행에 수십 분).
    # Rust 기반 fast 프로세서로 바꾸면 같은 전처리를 훨씬 빠르게 수행한다.
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    ).to(device)

    # peft: a single string target_modules is treated as a regex matched
    # against each submodule's full dotted name. This is what actually lets
    # us restrict LoRA to modules under "language_model" (and optionally
    # "vision_tower") instead of a literal "*" which matches nothing.
    def module_regex(prefix, names):
        alt = "|".join(names)
        return rf".*{prefix}.*\.({alt})$"

    target_regex = module_regex("language_model", LANGUAGE_TARGET_MODULES)
    if args.tune_vision:
        target_regex += "|" + module_regex("vision_tower", VISION_TARGET_MODULES)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_regex,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed so grads flow into LoRA through the frozen embedding layer

    # --- validation 분리: 모델/하이퍼파라미터 선택(early stopping, threshold)에만 쓰고
    # test.jsonl은 건드리지 않는다. match_cluster_id 단위로 묶어 같은 이미지/텍스트를
    # 공유하는 표본이 train/val에 걸치지 않게 하고, 라벨 비율은 층화로 유지한다. ---
    all_train_records = [json.loads(l) for l in open(args.train_jsonl, "r", encoding="utf-8")]
    if args.val_frac > 0:
        train_records, val_records = split_train_val(all_train_records, args.val_frac, args.seed)
        print(f"validation 분리: train {len(train_records)} / val {len(val_records)} "
              f"(val_frac={args.val_frac}, seed={args.seed})")
    else:
        train_records, val_records = all_train_records, []

    # --- stage 1: 전체 데이터 (박나현/조현서/정미정 세 조건 + non_hate 전부) ---
    use_aux = args.aux_text_weight > 0 or args.aux_image_weight > 0
    dataset = HateVLMDataset(train_records, args.dataset_root, split="train",
                              modality_dropout=modality_dropout)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=build_collate_fn(processor, build_aux=use_aux),
    )
    val_loader = None
    if val_records:
        val_dataset = HateVLMDataset(val_records, args.dataset_root, split="val")
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=build_collate_fn(processor, build_aux=False),  # validation loss는 항상 본 태스크 기준
        )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, loader, args.lr, args.epochs, args.grad_accum,
        vision_lr_scale=args.vision_lr_scale if args.tune_vision else 1.0,
    )

    hard_example_weights = {}
    n_matched = 0
    if args.hard_example_ids:
        with open(args.hard_example_ids, encoding="utf-8") as f:
            hard_ids = [line.strip() for line in f if line.strip()]
        hard_example_weights = {sid: args.hard_example_weight for sid in hard_ids}
        train_ids = {r["id"] for r in train_records}
        n_matched = sum(1 for sid in hard_ids if sid in train_ids)
        print(f"[hard-example] {len(hard_ids)}개 로드, train set과 {n_matched}개 매칭 "
              f"(가중치 {args.hard_example_weight}x)")
        if n_matched == 0:
            print("[hard-example][경고] 하나도 안 매칭됨 -- id 형식이나 train.jsonl이 맞는지 확인하세요.")

    if use_aux:
        print(f"[aux loss] 활성화: aux_text_weight={args.aux_text_weight}, "
              f"aux_image_weight={args.aux_image_weight} (stage 1 전용, stage 2에는 미적용)")

    class_weights = None
    if args.class_balance:
        # 정적 정답(r["target"]) 기준으로 계산한다. modality dropout이 일부 표본의 정답을
        # 학습 중에 "비혐오"로 바꾸지만(image_hate_text_safe), 그 비율은 작아서 근사로 충분하다.
        target_counts = Counter(r["target"] for r in train_records)
        n_total = sum(target_counts.values())
        class_weights = {t: n_total / (len(target_counts) * c) for t, c in target_counts.items()}
        print(f"[class-balance] 정답 분포 {dict(target_counts)} -> 가중치 "
              f"{ {t: round(w, 3) for t, w in class_weights.items()} }")

    n_stages = 2 if args.case_epochs > 0 else 1
    best_dir = os.path.join(args.output_dir, "best") if val_loader is not None else None
    print(f"== stage 1/{n_stages}: 전체 데이터 학습 ({len(dataset)} examples, {args.epochs} epochs) ==")
    step, best_val_loss = train_loop(
        model, loader, optimizer, scheduler, args.epochs, args.grad_accum,
        args.focal_gamma, args.focal_alpha, device, phase_name="stage1/전체",
        label_smoothing=args.label_smoothing, val_loader=val_loader, processor=processor,
        early_stopping_patience=args.early_stopping_patience, best_checkpoint_dir=best_dir,
        hard_example_weights=hard_example_weights,
        aux_text_weight=args.aux_text_weight, aux_image_weight=args.aux_image_weight,
        class_weights=class_weights,
    )

    # early stopping/체크포인트 저장은 val_loss가 가장 낮았던 시점의 가중치를 best_dir에
    # 저장해두는 것뿐이고, 지금 메모리에 있는 model은 그냥 "마지막 epoch" 상태다(그 사이
    # 과적합이 진행됐을 수 있음). best_dir가 있으면 반드시 그 가중치를 다시 불러와서
    # model을 best 상태로 되돌린 뒤 stage 2/최종 저장을 진행해야, early stopping이 실제로
    # 의미가 있다.
    if best_dir is not None and os.path.exists(os.path.join(best_dir, "adapter_model.safetensors")):
        best_state = load_file(os.path.join(best_dir, "adapter_model.safetensors"))
        # PEFT가 save_pretrained()로 저장한 safetensors 키에는 어댑터 이름("default")이
        # 빠져 있어서(예: "...lora_A.weight"), model.load_state_dict()로 바로 부으면
        # 실제 model.state_dict() 키("...lora_A.default.weight")와 거의 하나도 안 맞는다
        # -- strict=False라 예외 없이 조용히 "거의 다 unexpected/missing"으로 끝나고
        # 가중치는 사실상 하나도 안 바뀐다(마지막 epoch 가중치가 그대로 남음). PEFT
        # 전용 로더 set_peft_model_state_dict()가 어댑터 이름 매핑을 대신 처리해준다.
        from peft import set_peft_model_state_dict
        load_result = set_peft_model_state_dict(model, best_state)
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        print(f"[체크포인트 복원] best_dir({best_dir})의 가중치(val_loss={best_val_loss:.4f})를 다시 불러왔습니다 "
              f"(missing={len(missing)}, unexpected={len(unexpected)}).")
        if len(unexpected) >= len(best_state):
            print("[체크포인트 복원][경고] unexpected key 개수가 저장된 전체 키 개수와 같습니다 -- "
                  "복원이 실제로 안 됐을 수 있으니 확인하세요.")

    # --- stage 2 (선택): 담당 케이스 쌍만 다시 걸러서 이어서 추가 학습 ---
    # 예: case_label="hate" -> 박나현 케이스("유해 텍스트+유해 이미지 -> 유해"),
    # case_label="image_hate_text_safe" -> 조현서 케이스, "text_hate_image_safe" -> 정미정 케이스.
    # stage 1에서 나온 가중치를 그대로 이어받아(모델을 새로 만들지 않음) 계속 학습하므로,
    # 전체 데이터로 배운 일반적인 판단 능력 위에 이 케이스에 대한 성능을 더 강화하는 것이지
    # 전체 학습을 대체하는 게 아닙니다.
    # stage 1의 validation 분리(train_records, val_records 참고)와 같은 기준으로,
    # stage 2도 train_records(=val을 뺀 부분)에서만 서브셋을 고른다 -- 그러지 않으면
    # val 표본이 stage 2에서 다시 학습에 쓰여, 뒤이은 threshold 선택이 새 데이터가
    # 아니라 이미 본 데이터로 골라지는 누수가 생긴다.
    case_size = 0
    if args.case_epochs > 0:
        case_dataset = HateVLMDataset(train_records, args.dataset_root, label_filter=args.case_label, split="train")
        case_size = len(case_dataset)
        assert case_size > 0, (
            f"--case-label {args.case_label}에 해당하는 데이터가 {args.train_jsonl}에 없습니다."
        )
        if args.case_balance:
            # case_label이 "non_hate"면 반대는 "hate", 그 외(=hate, image_hate_text_safe,
            # text_hate_image_safe 등 최종 라벨이 유해인 모든 케이스)는 반대를 "non_hate"로 씀.
            other_label = "hate" if args.case_label == "non_hate" else "non_hate"
            other_dataset = HateVLMDataset(train_records, args.dataset_root, label_filter=other_label, split="train")
            assert len(other_dataset) > 0, (
                f"--case-balance를 쓰려면 반대 라벨({other_label}) 데이터도 {args.train_jsonl}에 있어야 합니다."
            )
            k = min(case_size, len(other_dataset))
            case_dataset.records = case_dataset.records + random.sample(other_dataset.records, k)
            random.shuffle(case_dataset.records)
            case_size = len(case_dataset)
            print(f"--case-balance: {other_label} {k}개를 섞어 stage 2를 균형 데이터로 구성 "
                  f"(총 {case_size}개)")
        case_loader = DataLoader(
            case_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=build_collate_fn(processor, build_aux=False),
        )
        case_lr = args.case_lr if args.case_lr is not None else args.lr / 2
        case_optimizer, case_scheduler = build_optimizer_and_scheduler(
            model, case_loader, case_lr, args.case_epochs, args.grad_accum,
        )
        print(f"\n== stage 2/{n_stages}: 담당 케이스(label={args.case_label}) 추가 학습 "
              f"({case_size} examples, {args.case_epochs} epochs, lr={case_lr}) ==")
        step, _ = train_loop(
            model, case_loader, case_optimizer, case_scheduler, args.case_epochs, args.grad_accum,
            args.focal_gamma, args.focal_alpha, device, phase_name=f"stage2/case={args.case_label}",
            step_offset=step, label_smoothing=args.label_smoothing,
            hard_example_weights=hard_example_weights,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter to {args.output_dir}")

    # validation에서 hate-score 임계값을 한 번만 고르고 run_info.json에 저장한다.
    # evaluate.py --threshold로 이 값을 test에 그대로(고정) 적용해야 검증→평가 임계값
    # 선택 원칙(6장)을 지킨다.
    selected_threshold, val_f1 = (None, None)
    if val_records:
        selected_threshold, val_f1 = select_threshold_on_val(
            model, processor, val_records, args.dataset_root, device,
            strategy=args.threshold_strategy, min_precision=args.min_precision, max_fpr=args.max_fpr,
        )

    # 팀 컨벤션(생성 기록): 재현/비교를 위해 이번 실행의 설정과 데이터 크기를
    # run_info.json으로 함께 남깁니다. 평가 지표는 evaluate.py가 별도로 채점하므로
    # 여기서는 학습 설정만 기록합니다.
    run_info = {
        "model_id": MODEL_ID,
        "train_jsonl": args.train_jsonl,
        "stage1_train_size": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "tune_vision": args.tune_vision,
        "vision_lr_scale": args.vision_lr_scale if args.tune_vision else None,
        "grad_checkpointing": args.grad_checkpointing,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "label_smoothing": args.label_smoothing,
        "modality_dropout": modality_dropout,
        "modality_dropout_target": "text_only",  # 이미지를 가리면 정답도 텍스트 단독 기준으로 바뀜
        "class_balance": args.class_balance,
        "class_weights": class_weights,
        "threshold_strategy": args.threshold_strategy,
        "min_precision": args.min_precision if args.threshold_strategy == "precision_constrained" else None,
        "max_fpr": args.max_fpr if args.threshold_strategy == "precision_constrained" else None,
        "hard_example_ids_file": args.hard_example_ids,
        "hard_example_weight": args.hard_example_weight if args.hard_example_ids else None,
        "n_hard_examples_matched": n_matched if args.hard_example_ids else 0,
        "aux_text_weight": args.aux_text_weight,
        "aux_image_weight": args.aux_image_weight,
        "case_label": args.case_label,
        "case_epochs": args.case_epochs,
        "case_lr": (args.case_lr if args.case_lr is not None else args.lr / 2) if args.case_epochs > 0 else None,
        "case_balance": args.case_balance,
        "stage2_train_size": case_size,
        "seed": args.seed,
        "image_size": IMAGE_SIZE,
        "val_frac": args.val_frac,
        "val_size": len(val_records),
        "early_stopping_patience": args.early_stopping_patience,
        "best_val_loss": best_val_loss if val_records else None,
        "selected_threshold": selected_threshold,
        "val_f1_at_threshold": val_f1,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print(f"run_info.json saved to {args.output_dir}")


if __name__ == "__main__":
    main()
