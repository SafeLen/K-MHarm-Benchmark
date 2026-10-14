"""
모델별로 달라지는 부분"만" 모아둔 파일.

train_lora.py / evaluate.py는 --model 옵션 하나로 여기서 어댑터를 받아 쓰고,
학습 루프 / 채점 방식 / LoRA 설정 / 임계값 선택 / 지표 계산은 전부 공통 코드다.

모델마다 다른 것 (= 그 모델에 딸린 부분이라 달라도 공정성에 문제 없음)
    - 불러오는 클래스, processor, chat template 적용 방식
공통으로 고정한 것
    - 입력 이미지: image_prep.py로 똑같이 384x384로 만든 뒤 각 모델 processor에 넘김
    - 프롬프트 문구, 정답 토큰("혐오"/"비혐오"), 정답 토큰 뒤에 아무것도 붙이지 않음
    - LoRA: 언어모델의 q/k/v/o/gate/up/down_proj 에만, 비전 인코더/프로젝터는 동결
    - 채점: 정답 토큰의 teacher-forced log-likelihood (아래 target_logprob)
"""
import os
import tempfile

import torch
import torch.nn.functional as F

MODEL_REGISTRY = {
    "varco": {
        "hf_id": "NCSOFT/VARCO-VISION-2.0-1.7B",
        "display": "VARCO-VISION-2.0-1.7B",
    },
    "hyperclovax": {
        "hf_id": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
        "display": "HyperCLOVAX-SEED-Vision-3B",
    },
    "bllossom": {
        "hf_id": "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
        "display": "Bllossom-AICA-5B",
    },
    # --- 2차 비교 (더 큰 8B급: 한국어 특화 1 + 범용 1) ---
    "ax": {  # SK텔레콤, Qwen2 기반 언어모델 + SigLIP, trust_remote_code 필요 (transformers >= 4.49)
        "hf_id": "skt/A.X-4.0-VL-Light",
        "display": "A.X-4.0-VL-Light-8B",
    },
    "qwen3vl": {  # Alibaba 범용 VLM, transformers >= 4.57 필요
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "display": "Qwen3-VL-8B-Instruct",
    },
    # --- Zero-shot 평가용 범용 모델 (transformers 기본 탑재 HF 포맷) ---
    "internvl3": {  # OpenGVLab, transformers >= 4.52
        "hf_id": "OpenGVLab/InternVL3-8B-hf",
        "display": "InternVL3-8B",
    },
    "llavaov": {  # LLaVA-OneVision 7B (VARCO와 같은 구조의 범용 모델)
        "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "display": "LLaVA-OneVision-7B",
    },
}

LORA_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# 이 이름이 경로에 들어간 모듈은 비전 쪽이므로 LoRA 대상에서 제외
VISION_PREFIXES = ("vision_tower", "vision_model", "visual", "vision_encoder",
                   "multi_modal_projector", "mm_projector", "image_newline")

DTYPE = torch.bfloat16


def to_device(obj, device, dtype=DTYPE):
    """dict/list/tensor를 재귀적으로 device로 옮기고, 실수형 텐서는 bf16으로 맞춘다."""
    if torch.is_tensor(obj):
        obj = obj.to(device)
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict) or hasattr(obj, "items"):
        return {k: to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device, dtype) for v in obj)
    return obj


class BaseAdapter:
    key = None

    def __init__(self, key, device):
        self.key = key
        self.cfg = MODEL_REGISTRY[key]
        self.hf_id = self.cfg["hf_id"]
        self.display = self.cfg["display"]
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None

    # --- 모델별로 구현 ---
    def load(self):
        raise NotImplementedError

    def prompt_inputs(self, image, prompt):
        """이미지 + 사용자 프롬프트 -> 생성 직전(assistant 차례)까지의 모델 입력 dict (batch=1)."""
        raise NotImplementedError

    # --- 공통 ---
    def lora_target_modules(self, model):
        """언어모델 쪽 선형층의 '전체 이름' 목록. 모델마다 경로 이름이 달라도
        '언어모델의 attention/MLP 투영층'이라는 같은 위치에 붙게 된다."""
        names = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if name.split(".")[-1] not in LORA_TARGET_NAMES:
                continue
            if any(p in name for p in VISION_PREFIXES):
                continue
            names.append(name)
        assert names, f"{self.key}: LoRA를 붙일 언어모델 층을 찾지 못했습니다."
        return names

    def target_ids(self, target):
        return self.tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    def append_target(self, inputs, target):
        """프롬프트 입력 뒤에 정답 토큰을 이어붙인다 (모든 모델 동일 방식)."""
        tid = self.target_ids(target)
        k = tid.shape[0]
        out = dict(inputs)
        L = inputs["input_ids"].shape[1]
        out["input_ids"] = torch.cat([inputs["input_ids"], tid[None].to(inputs["input_ids"].device)], dim=1)
        out["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones((1, k), dtype=inputs["attention_mask"].dtype,
                                                  device=inputs["attention_mask"].device)], dim=1)
        # 시퀀스 길이 축을 가진 보조 마스크(예: Mllama의 cross_attention_mask)도 같이 늘려준다
        for key, v in inputs.items():
            if key in ("input_ids", "attention_mask") or not torch.is_tensor(v):
                continue
            if v.dim() >= 2 and v.shape[1] == L and ("mask" in key or "token_type" in key):
                tail = v[:, -1:].expand(*([v.shape[0], k] + list(v.shape[2:])))
                out[key] = torch.cat([v, tail], dim=1)
        return out, tid


class VarcoAdapter(BaseAdapter):
    def load(self):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.hf_id, torch_dtype=DTYPE, attn_implementation="sdpa").to(self.device)
        return self.model

    def prompt_inputs(self, image, prompt):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        enc = self.processor(text=text, images=image, return_tensors="pt")
        return to_device(dict(enc), self.device)


class BllossomAdapter(BaseAdapter):
    def load(self):
        from transformers import AutoProcessor, MllamaForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.hf_id, torch_dtype=DTYPE, attn_implementation="sdpa").to(self.device)
        return self.model

    def prompt_inputs(self, image, prompt):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        # 모델 카드 예시대로 chat template이 BOS를 이미 넣으므로 add_special_tokens=False
        enc = self.processor(image, text, add_special_tokens=False, return_tensors="pt")
        return to_device(dict(enc), self.device)


class HyperClovaXAdapter(BaseAdapter):
    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=DTYPE, trust_remote_code=True).to(self.device)
        return self.model

    def prompt_inputs(self, image, prompt):
        # 모델 카드 형식: image는 경로/URL로 넘긴다 -> 전처리된 이미지를 임시 파일로 저장해서 전달
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.save(path)
            chat = [
                # chat template이 filename을 프롬프트 텍스트에 그대로 넣으므로, 임시 파일의 무작위 이름 대신 고정 이름을 쓴다
                {"role": "user", "content": [{"type": "image", "image": path, "filename": "image.png"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            enc = self.processor.apply_chat_template(
                chat, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
        finally:
            os.remove(path)
        return to_device(dict(enc), self.device)


def keep_forward_args(model, enc):
    """processor가 만든 키 중 모델 forward가 받지 않는 것(예: image_sizes)을 뺀다.
    forward가 **kwargs를 받으면 그대로 둔다."""
    import inspect
    params = inspect.signature(model.forward).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return enc
    return {k: v for k, v in enc.items() if k in params}


class AXAdapter(BaseAdapter):
    """skt/A.X-4.0-VL-Light (모델 카드 사용법 그대로: AutoModelForCausalLM + AutoProcessor, trust_remote_code)."""

    def load(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        # A.X의 processing_ax4vl.py가 transformers 4.53에서 삭제된 내부 함수를 import한다.
        # 우리는 processor(images=..., text=...)로 키워드 지정 호출만 하므로 순서를 그대로 돌려주는 함수로 대체.
        import transformers.processing_utils as _pu
        if not hasattr(_pu, "_validate_images_text_input_order"):
            _pu._validate_images_text_input_order = lambda images=None, text=None: (images, text)
        self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        # config에 flash_attention_2가 지정돼 있어 flash-attn 없이도 돌도록 sdpa(안 되면 eager)로 바꾼다.
        # 어텐션 구현 방식만 다를 뿐 계산 결과는 같다.
        last_err = None
        for impl in ("sdpa", "eager"):
            cfg = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True)
            for sub in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None)):
                if sub is not None:
                    try:
                        sub._attn_implementation = impl
                    except Exception:
                        pass
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_id, config=cfg, torch_dtype=DTYPE, trust_remote_code=True).to(self.device)
                print(f"[{self.display}] attention: {impl}")
                break
            except (ValueError, ImportError) as e:
                last_err = e
        else:
            raise last_err
        return self.model

    def prompt_inputs(self, image, prompt):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = self.processor(images=[image], text=[text], padding=True, return_tensors="pt")
        return to_device(keep_forward_args(self.model, dict(enc)), self.device)


class Qwen3VLAdapter(BaseAdapter):
    """Qwen/Qwen3-VL-8B-Instruct (transformers 기본 탑재 클래스)."""

    def load(self):
        import transformers
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(f"Qwen3-VL은 transformers>=4.57이 필요합니다 (현재 {transformers.__version__}).") from e
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.hf_id, torch_dtype=DTYPE, attn_implementation="sdpa").to(self.device)
        return self.model

    def prompt_inputs(self, image, prompt):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        return to_device(dict(enc), self.device)


class LlavaOVAdapter(VarcoAdapter):
    """llava-hf/llava-onevision-qwen2-7b-ov-hf. VARCO와 같은 LlavaOnevision 클래스라 그대로 상속."""


class InternVL3Adapter(BaseAdapter):
    """OpenGVLab/InternVL3-8B-hf (transformers 기본 탑재 HF 포맷, AutoModelForImageTextToText)."""

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id, torch_dtype=DTYPE, attn_implementation="sdpa").to(self.device)
        return self.model

    def prompt_inputs(self, image, prompt):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        return to_device(dict(enc), self.device)


ADAPTERS = {"varco": VarcoAdapter, "hyperclovax": HyperClovaXAdapter, "bllossom": BllossomAdapter,
            "ax": AXAdapter, "qwen3vl": Qwen3VLAdapter, "internvl3": InternVL3Adapter, "llavaov": LlavaOVAdapter}
MODEL_KEYS = list(ADAPTERS)


def get_adapter(key, device):
    if key not in ADAPTERS:
        raise ValueError(f"--model은 {list(ADAPTERS)} 중 하나여야 합니다: {key}")
    return ADAPTERS[key](key, device)


def forward_tail_logits(model, inputs, n_tail):
    """마지막 n_tail개 위치의 logits만 반환. 지원하는 모델은 logits_to_keep으로 메모리를 아낀다.
    이미지 토큰이 모델 내부에서 확장돼도, 정답 토큰은 항상 맨 뒤에 있으므로 뒤에서부터 자르면 정확하다."""
    try:
        out = model(**inputs, use_cache=False, logits_to_keep=n_tail)
    except TypeError:
        out = model(**inputs, use_cache=False)
    return out.logits[:, -n_tail:, :]


def target_logprob(model, inputs_with_target, target_ids):
    """정답 토큰들의 log-likelihood 합 (teacher forcing). 학습 loss와 채점에 똑같이 쓴다."""
    k = target_ids.shape[0]
    logits = forward_tail_logits(model, inputs_with_target, k + 1)[0, :-1, :].float()  # (k, V)
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, target_ids.to(logp.device)[:, None]).sum()


@torch.no_grad()
def hate_score(adapter, model, image, prompt):
    """score = logp("혐오") - logp("비혐오"). 클수록 혐오 쪽. 프롬프트 인코딩은 한 번만 한다."""
    from common import HATE_TOKEN, NONHATE_TOKEN
    base = adapter.prompt_inputs(image, prompt)
    inp_h, tid_h = adapter.append_target(base, HATE_TOKEN)
    inp_n, tid_n = adapter.append_target(base, NONHATE_TOKEN)
    lp_h = target_logprob(model, inp_h, tid_h).item()
    lp_n = target_logprob(model, inp_n, tid_n).item()
    return lp_h - lp_n, lp_h, lp_n


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
