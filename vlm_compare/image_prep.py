"""
공통 이미지 전처리 (연구 설계서 4장 "입력 전처리 원칙 - 이미지" 반영).

train_lora.py와 evaluate.py가 공유해서 쓴다:
    - EXIF orientation 반영, RGB 변환, alpha 채널 제거는 train/val/test 공통.
    - train: RandomResizedCrop(scale=0.90~1.00) + 약한 ColorJitter. 좌우 반전은
      OCR 결과나 방향성 있는 기호를 바꿀 수 있어 기본 제외.
    - validation/test: Resize + CenterCrop만 적용, 확률적 augmentation 없음.
    - 손상 파일은 예외를 잡아 sample_id와 사유를 기록하고 호출부에서 건너뛸 수 있게
      None을 반환한다.

VARCO-VISION은 LLaVA-OneVision의 anyres 전처리를 쓰므로, 최종적으로는 여기서 만든
정사각형 PIL 이미지를 processor()에 넘겨 모델 자체의 224 정규화/타일링을 타게 한다.
"""
import os

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from torchvision import transforms
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

IMAGE_SIZE = 384  # train_lora.py/evaluate.py의 anyres 단일 타일 크기와 일치시켜야 함

_corrupt_log = []


def load_image_for_model(path, sample_id=None):
    """EXIF orientation 반영 + RGB 변환 + alpha 제거까지 마친 PIL.Image를 반환.
    손상 파일이면 None을 반환하고 (sample_id, path, 사유)를 _corrupt_log에 남긴다."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # EXIF orientation 반영
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError) as e:
        _corrupt_log.append({"sample_id": sample_id, "path": path, "reason": str(e)})
        print(f"[경고] 손상된 이미지 파일, 건너뜁니다: {path} ({e})")
        return None


def get_corrupt_log():
    return list(_corrupt_log)


def build_train_transform(image_size=IMAGE_SIZE):
    """train: RandomResizedCrop(scale=0.90~1.00) + 약한 ColorJitter. 좌우 반전 없음."""
    if not _HAS_TORCHVISION:
        return None
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.90, 1.00)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    ])


def build_eval_transform(image_size=IMAGE_SIZE):
    """validation/test: Resize + CenterCrop만, 확률적 augmentation 없음."""
    if not _HAS_TORCHVISION:
        return None
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])


def prepare_image(path, split, sample_id=None, image_size=IMAGE_SIZE):
    """split in {"train", "val", "test"}. torchvision이 없으면 단순 resize로 대체
    (Colab 환경에는 기본 포함되어 있어 보통 문제되지 않음)."""
    img = load_image_for_model(path, sample_id=sample_id)
    if img is None:
        return None
    if not _HAS_TORCHVISION:
        return img.resize((image_size, image_size))
    tfm = build_train_transform(image_size) if split == "train" else build_eval_transform(image_size)
    return tfm(img)
