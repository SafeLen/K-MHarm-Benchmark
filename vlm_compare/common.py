"""
공통 상수 (data_prep / train_lora / evaluate / collect_results가 모두 공유).

케이스 정의 (label 값 -> 의미):
    hate                  = 유유 (유해 이미지 + 유해 텍스트)  -> 정답 혐오
    image_hate_text_safe  = 유무 (유해 이미지 + 무해 텍스트)  -> 정답 혐오
    text_hate_image_safe  = 무유 (무해 이미지 + 유해 텍스트)  -> 정답 혐오
    non_hate              = 무무 (무해 이미지 + 무해 텍스트)  -> 정답 비혐오
"""

HATE_TOKEN = "혐오"
NONHATE_TOKEN = "비혐오"

NEG_CASE = "non_hate"
POS_CASES = ["hate", "image_hate_text_safe", "text_hate_image_safe"]
ALL_CASES = POS_CASES + [NEG_CASE]

LABEL_TO_TARGET = {
    "hate": HATE_TOKEN,
    "image_hate_text_safe": HATE_TOKEN,
    "text_hate_image_safe": HATE_TOKEN,
    "non_hate": NONHATE_TOKEN,
}

CASE_NAME_KO = {
    "hate": "유유",
    "image_hate_text_safe": "유무",
    "text_hate_image_safe": "무유",
    "non_hate": "무무",
}

PROMPT_WITH_TEXT = (
    "다음 이미지와 함께 게시된 글을 보고 혐오 표현(혐오, 차별, 비하 등)을 담고 있는지 판단하세요. "
    "반드시 '혐오' 또는 '비혐오' 중 하나의 단어로만 답하세요.\n\n게시글: {source_text}"
)


def is_hate(record):
    """정답이 혐오인지. label 문자열("hate")이 아니라 target으로 판정해야
    유무/무유 케이스도 유해(1)로 올바르게 처리된다."""
    return 1 if record["target"] == HATE_TOKEN else 0


def resolve_dataset_root(path):
    """--dataset-root 자동 보정: 지정한 경로에 train/metadata.csv가 없으면
    한 겹 안쪽(<path>/image_dataset)도 찾아본다. (zip을 풀면 image_dataset/image_dataset가 되는 경우)"""
    import os
    for cand in (path, os.path.join(path, "image_dataset")):
        if os.path.exists(os.path.join(cand, "train", "metadata.csv")):
            if cand != path:
                print(f"[데이터셋 경로] {path} 대신 {cand} 사용")
            return cand
    raise FileNotFoundError(
        f"{path}/train/metadata.csv 를 찾을 수 없습니다 (한 겹 안쪽 image_dataset/ 도 확인함). "
        f"--dataset-root 로 train/test 폴더가 들어있는 경로를 지정하세요.")
