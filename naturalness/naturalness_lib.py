"""
자연도 점수 (논문 4.3, 방법론 3.7.2)

학습을 마친 VARCO-VISION-2.0-1.7B + LoRA 모델 자체를 판정자로 쓴다(외부 모델 없음).
이미지와 글이 하나의 게시물로 함께 올라올 법한지 묻고 '자연' 또는 '어색' 한 단어로만
답하게 한 뒤, 두 단어 첫 토큰의 teacher-forcing 로그 확률 차이를 점수로 쓴다.

    naturalness = log p('자연'의 첫 토큰 | 입력) - log p('어색'의 첫 토큰 | 입력)   (클수록 자연스러움)

글을 생성하지 않으므로 출력 형식 오류가 없다. 모델, 점수, 임계값, 정답은 바꾸지 않는
읽기 전용 단계다.

주의: 원래 실험의 프롬프트 문구는 남아 있지 않아 방법론 3.7.2의 서술대로 다시 작성했다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(HERE, "..", "main_experiment", "shared")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

MODEL_ID = "NCSOFT/VARCO-VISION-2.0-1.7B"
NATURAL_TOKEN = "자연"
AWKWARD_TOKEN = "어색"
NAT_PROMPT = (
    "다음 이미지와 함께 게시된 글이 실제로 하나의 게시물로 함께 올라올 법한지 판단하세요. "
    "반드시 '자연' 또는 '어색' 중 하나의 단어로만 답하세요.\n\n게시글: {source_text}"
)


# 교차 결합 조건: (label 이름, 이미지를 가져올 원본 label, 텍스트를 가져올 원본 label)
CROSS_CASES = {
    "image_hate_text_safe": ("hate", "non_hate"),   # 조건 10
    "text_hate_image_safe": ("non_hate", "hate"),   # 조건 01
}


def natural_records(dataset_root, split):
    """data_prep.convert_split과 같은 방식으로 원본 표본(조건 11·비혐오)만 만든다."""
    import data_prep

    records = []
    for row in data_prep.read_rows(dataset_root, split):
        label = row["label"].strip()
        if label not in ("hate", "non_hate"):
            continue
        if not os.path.exists(os.path.join(dataset_root, row["image_path"])):
            raise FileNotFoundError(row["image_path"])
        records.append(data_prep.make_record(
            row["id"], row["image_path"], label, row.get("source_text", "") or "",
            include_text=True, mask_names=False,
            hate_categories=row.get("hate_categories", "") or "",
            image_source=row.get("image_source", "unknown") or "unknown",
        ))
    return records


class NaturalnessScorer:
    """한 seed의 LoRA 어댑터로 (이미지, 글) 쌍의 자연도 점수를 매긴다.
    같은 이미지를 여러 후보 글과 채점하므로(3.7.4) 전처리한 이미지는 캐시한다."""

    def __init__(self, adapter_dir, dataset_root, device=None):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        import image_prep

        self.torch = torch
        self.image_prep = image_prep
        self.dataset_root = dataset_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # evaluate.py와 같은 이유로 bf16(cuda) / fp32(cpu), use_fast=True
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(adapter_dir, use_fast=True)
        base = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=self.dtype, attn_implementation="sdpa").to(self.device)
        self.model = PeftModel.from_pretrained(base, adapter_dir).to(self.device)
        self.model.eval()

        tok = self.processor.tokenizer
        self.natural_id = tok(NATURAL_TOKEN, add_special_tokens=False)["input_ids"][0]
        self.awkward_id = tok(AWKWARD_TOKEN, add_special_tokens=False)["input_ids"][0]
        assert self.natural_id != self.awkward_id, "'자연'과 '어색'의 첫 토큰이 같습니다"
        self._images = {}

    def image(self, image_path):
        if image_path not in self._images:
            img = self.image_prep.prepare_image(os.path.join(self.dataset_root, image_path), "test",
                                                sample_id=image_path)
            if img is None:
                raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
            self._images[image_path] = img
        return self._images[image_path]

    def score(self, image_path, text):
        torch = self.torch
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": NAT_PROMPT.format(source_text=text)},
        ]}]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        enc = self.processor(text=prompt_text, images=self.image(image_path), return_tensors="pt")
        kwargs = {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "pixel_values": enc["pixel_values"].to(self.device, self.dtype),
        }
        if "image_sizes" in enc:
            kwargs["image_sizes"] = enc["image_sizes"].to(self.device)
        with torch.no_grad():
            logits = self.model(**kwargs).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        return float(logp[self.natural_id] - logp[self.awkward_id])
