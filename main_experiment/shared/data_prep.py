"""
image_dataset/{train,test}/metadata.csv -> hate_pipeline_varco/data/{split}.jsonl

Each output record is a VLM instruction-tuning example:
    {
        "id": ...,
        "image_path": "train/non_hate/xxxx.png",   # relative to --dataset-root
        "label": "hate" | "non_hate" | "image_hate_text_safe" | "text_hate_image_safe",
        "source_text": "...",                       # NFKC-normalized, 은어/자모 보존
        "source_text_raw": "...",                    # 정규화 전 원문 (감사용 보존)
        "image_source": "...",                       # 생성기/출처 표기 (레이블과 무관한
                                                       # 생성기 특성 학습 여부 점검용)
        "hate_categories": "...",                     # 원본 metadata.csv의 세부 유형 (보존)
        "group_id": "...",                            # 동일/유사 표본 묶음 -> split 누수 방지 확인용
        "prompt": "<Korean instruction asking hate/non-hate>",
        "target": "혐오" | "비혐오"
    }

Run:
    python data_prep.py --dataset-root ../image_dataset --out-dir data --case-pairs

이 파일은 박나현/조현서/정미정 세 사람이 공유하는 단일 data_prep.py다(shared/ 아래 1벌만
존재). 세 사람이 각자 담당하던 "케이스"는 세 개의 서로 다른 스크립트가 아니라, 아래
LABEL_TO_TARGET에 정의된 label 값으로 구분된다:
    - hate                 : 박나현 케이스 (이미지=유해, 텍스트=유해). 원본 metadata.csv의
                              hate 라벨을 그대로 쓴다(augmentation 불필요 -- 원래도 이미지·
                              텍스트가 함께 유해로 스크랩된 게시글이므로).
    - image_hate_text_safe : 조현서 케이스 (이미지=유해, 텍스트=무해). --case-pairs로
                              hate 이미지 x non_hate 텍스트를 크로스 페어링해서 생성.
    - text_hate_image_safe : 정미정 케이스 (이미지=무해, 텍스트=유해). --case-pairs로
                              hate 텍스트 x non_hate 이미지를 크로스 페어링해서 생성
                              (mijung이 노트북에서 수동으로 하던 build_hate_text_nonhate_image를
                              흡수 -- 단, 원본은 train split에만 적용되던 제약이 있었는데
                              여기서는 image_hate_text_safe와 동일하게 train/test 양쪽에
                              적용한다. 평가 단계에서 --eval-filter로 이 케이스만 따로
                              채점하려면 test에도 있어야 하기 때문).
    - (제외) 무해 이미지 + 무해 텍스트(00) 조합은 이번 연구 설계에 포함하지 않는다
      (analysis/matched_design_analysis.py 상단 docstring 참고).
--case-pairs 하나로 image_hate_text_safe/text_hate_image_safe 두 케이스를 모두 생성한다
(세 조건을 하나의 pool에서 매칭 생성해 하나의 train/val/test로 합치는 통합 설계이므로,
한쪽만 만들 이유가 없다).

전처리 원칙 (연구 설계서 4장 "입력 전처리 원칙" 반영):
    - 텍스트: Unicode NFKC 정규화 + 앞뒤 공백 제거 + 연속 공백 축소만 적용한다.
      자모 분해, 은어, 반복 글자(예: ㅋㅋㅋ)는 혐오 판단의 단서가 될 수 있으므로
      임의로 교정/삭제하지 않는다. 정규화 전 원문은 source_text_raw로 별도 보존한다.
    - 이미지: RGB 변환 + EXIF orientation 반영 + alpha 채널 제거는 학습/평가 스크립트
      (train_lora.py, evaluate.py)에서 이미지를 열 때 공통으로 적용한다(아래 image_prep.py
      의 load_image_for_model 참고). 여기서는 손상 파일만 이 단계에서 걸러 로그로 남긴다.
    - 실명 마스킹(<PERSON>) 여부는 팀 회의에서 확정하기 전까지 기본은 원문 유지이며,
      --mask-person을 켜면 model_text(=source_text/prompt에 들어가는 텍스트)에도
      이미지 생성 프롬프트와 동일한 규칙으로 <PERSON>을 적용한다. source_text_raw는
      마스킹 여부와 무관하게 항상 원문을 보존한다.

Optional FN/FP-targeted augmentation (train split only, --augment):
    - Slang/변형어 augmentation (FN 대응): 혐오 문장 속 원형 단어를
      은어/변형어로 치환한 사본을 추가로 만들어, "틀니딱딱충" 같이 학습
      데이터에 없던 파생형도 혐오로 학습하게 함.
    - Counterfactual augmentation (FP 대응): "다문화", "성소수자", "일베" 같은
      민감 주제어가 포함되지만 실제로는 비혐오인 문장을, 기존 non_hate
      이미지 중 하나와 짝지어 새 예시로 추가. 모델이 주제어 자체가 아니라
      문맥/의도를 보도록 유도.
    --augment는 텍스트를 프롬프트에 넣어야 의미가 있으므로 --include-text를
    자동으로 함께 켭니다.

Case-pair generation (train+test 양쪽, --case-pairs):
    - image_hate_text_safe: hate 이미지 x non_hate 텍스트 크로스 페어링.
    - text_hate_image_safe: hate 텍스트 x non_hate 이미지 크로스 페어링.
    두 케이스 모두 매칭 설계(matched design)를 위해 group_id에 원본 이미지·텍스트 쪽
    group_id를 함께 새겨서, 분석 단계(analysis/matched_design_analysis.py)에서 "같은
    이미지/텍스트가 다른 조건에도 등장"하는 매칭 관계를 추적할 수 있게 한다.

Optional modality-variant export (--emit-modality-variants):
    Tier 2 Agent 라우팅(연구 설계서 5장)의 (0,0,1)/(1,1,0)/(1,0,0)/(0,1,0) 패턴을
    가리려면 같은 표본에 대해 text-only / image-only / multimodal 예측이 모두
    필요하다. joint(멀티모달)와 text-only(--blank-image로 이미지만 가림)는 기존
    evaluate.py 옵션만으로 나오므로, 여기서는 이미지 전용(prompt에서 텍스트를 뺀)
    변형만 추가로 {split}_image_only.jsonl 로 내보낸다.
"""
import argparse
import csv
import json
import os
import random
import unicodedata
from collections import Counter

PROMPT_IMAGE_ONLY = (
    "다음 이미지를 보고 혐오 표현(혐오, 차별, 비하 등)을 담고 있는지 판단하세요. "
    "반드시 '혐오' 또는 '비혐오' 중 하나의 단어로만 답하세요."
)

PROMPT_WITH_TEXT = (
    "다음 이미지와 함께 게시된 글을 보고 혐오 표현(혐오, 차별, 비하 등)을 담고 있는지 판단하세요. "
    "반드시 '혐오' 또는 '비혐오' 중 하나의 단어로만 답하세요.\n\n게시글: {source_text}"
)

# 팀 정의: 이미지·텍스트 둘 중 하나라도 유해면 최종 라벨은 유해(OR 규칙).
# 그래서 image_hate_text_safe/text_hate_image_safe도 target="혐오".
LABEL_TO_TARGET = {
    "hate": "혐오",
    "non_hate": "비혐오",
    "image_hate_text_safe": "혐오",   # 조현서 케이스 (10): augment_case_pairs()로 생성
    "text_hate_image_safe": "혐오",   # 정미정 케이스 (01): augment_case_pairs()로 생성
}

# --- FN 대응: 원형 단어 -> 은어/변형어 매핑. 필요에 따라 계속 추가하세요. ---
SLANG_MAP = {
    "노인": ["틀딱", "틀니딱딱충"],
    "전라도": ["홍어", "홍어새끼"],
    "외국인노동자": ["외노자"],
    "여성": ["김치녀", "보슬아치"],
    "남성": ["한남", "한남충"],
    "장애인": ["병신", "장애우충"],
    # 이번 라운드 FN 분석에서 새로 발견된 은어 (재학습 후에도 여전히 못 잡던 것)
    "자살": ["운지"],  # 노무현 전 대통령 조롱 은어, FN 사례("...기독교...운지하고...")에서 확인
}
MAX_SLANG_AUG_PER_RECORD = 3  # 한 원본 문장에서 생성할 변형어 사본 상한

# --- FP 대응: 민감 주제어가 있지만 비혐오인 문장들. 실제 non_hate 이미지와 짝지어 추가. ---
COUNTERFACTUAL_SENTENCES = {
    "다문화": [
        "다문화 가정 지원 정책이 확대되었다.",
        "다문화 축제에 많은 시민들이 참여했다.",
    ],
    "성소수자": [
        "성소수자 인권에 대한 토론회가 열렸다.",
        "성소수자 인권 보호를 위한 법안이 발의되었다.",
    ],
    "일베": [
        "일베 같은 혐오 사이트의 문제를 비판하는 기사가 나왔다.",
        "일베발 가짜뉴스에 대한 팩트체크 보도가 있었다.",
    ],
    "이주민": [
        "이주민 정착 지원 센터가 새로 문을 열었다.",
        "이주민 자녀를 위한 한국어 교육 프로그램이 시작됐다.",
    ],
    "장애인": [
        "장애인 이동권 개선을 위한 예산이 편성됐다.",
        "장애인 고용률이 전년 대비 상승했다.",
    ],
}

# 실명 마스킹(--mask-person)용 아주 단순한 규칙. 실제 실명 목록/개체명 인식이 아니라
# 팀이 마스킹 규칙을 확정하기 전까지 쓸 자리 표시자(placeholder)이며, 이미지 생성
# 프롬프트에서 쓰는 규칙과 반드시 동일하게 유지해야 두 모달리티의 전처리 기준이
# 어긋나지 않는다. 실제 규칙이 정해지면 이 함수만 교체하면 된다.
def mask_person(text):
    return text


def normalize_text(text):
    """공통 텍스트 전처리: NFKC 정규화 + 앞뒤 공백 제거 + 연속 공백 축소만 적용.
    자모 분해, 은어, 반복 글자(ㅋㅋㅋ 등)는 혐오 판단의 단서이므로 손대지 않는다."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = " ".join(text.split())
    return text


def read_rows(dataset_root: str, split: str):
    csv_path = os.path.join(dataset_root, split, "metadata.csv")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_record(row_id, image_path, label, source_text_raw, include_text, mask_names,
                 hate_categories="", image_source="unknown",
                 image_group_id=None, text_group_id=None):
    """image_group_id/text_group_id: 이 표본의 이미지/텍스트가 각각 "원래 어느 원본 표본
    출신인지"를 가리키는 키. 자연 표본(원본 metadata.csv 행 그대로)은 이미지와 텍스트가
    같은 행 출신이므로 둘 다 row_id로 기본값이 채워진다. augment_case_pairs()로 만든
    크로스 페어링 표본은 이미지와 텍스트가 서로 다른 원본 행에서 왔으므로 이 둘이
    달라진다 -- 이게 바로 "같은 이미지/텍스트가 다른 조건에도 등장"하는 매칭 관계의
    원천이며, 하나의 문자열로 합친 group_id로는 이 관계를 추적할 수 없어서(조건마다
    합친 문자열이 서로 다르므로) 두 필드를 분리해서 남긴다.
    assign_match_clusters()가 이 두 필드로부터 실제 random-effect/bootstrap 클러스터를
    계산한다."""
    source_text_norm = normalize_text(source_text_raw)
    model_text = mask_person(source_text_norm) if mask_names else source_text_norm
    prompt = (
        PROMPT_WITH_TEXT.format(source_text=model_text)
        if include_text
        else PROMPT_IMAGE_ONLY
    )
    return {
        "id": row_id,
        "image_path": image_path,
        "label": label,
        "source_text": model_text,
        "source_text_raw": source_text_raw,
        "hate_categories": hate_categories,
        "image_source": image_source,
        "image_group_id": image_group_id if image_group_id is not None else row_id,
        "text_group_id": text_group_id if text_group_id is not None else row_id,
        "prompt": prompt,
        "target": LABEL_TO_TARGET[label],
    }


class _UnionFind:
    """image_group_id/text_group_id를 노드로 하는 그래프에서 연결 요소(connected
    component)를 찾기 위한 간단한 union-find. 표본 하나가 (image_group_id, text_group_id)
    두 노드를 잇는 간선 하나를 만든다고 보면, 연결된 요소 전체가 "서로 얽힌 매칭 클러스터"
    (같은 이미지 또는 같은 텍스트를 공유하는 모든 표본들)가 된다."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def assign_match_clusters(records):
    """모든 표본에 match_cluster_id를 부여한다: image_group_id/text_group_id를 공유하는
    표본들(예: 어떤 hate 이미지의 자연 표본 + 그 이미지를 재사용한 image_hate_text_safe
    표본)은 전이적으로(transitively) 같은 match_cluster_id를 받는다.

    이 클러스터가 두 가지 용도에 쓰인다:
      1) train_lora.py의 split_train_val -- 같은 이미지/텍스트를 공유하는 표본이 train과
         val에 걸쳐 누수되지 않게 클러스터 단위로 나눈다.
      2) analysis/matched_design_analysis.py -- cluster bootstrap/random effect의 리샘플링
         단위. 조건(hate/image_hate_text_safe/text_hate_image_safe)이 달라도 같은
         이미지·텍스트를 공유하면 독립 표본이 아니므로, 이 클러스터를 통으로 리샘플해야
         한다.
    """
    uf = _UnionFind()
    for r in records:
        uf.union(("img", r["image_group_id"]), ("txt", r["text_group_id"]))
    for r in records:
        root_type, root_id = uf.find(("img", r["image_group_id"]))
        r["match_cluster_id"] = f"cluster_{root_type}_{root_id}"
    return records


def make_image_only_record(base_record):
    """joint/text-only 예측과 짝지어 Tier 2 라우팅용 이미지 단독 예측을 낼 수 있도록,
    동일 표본을 텍스트 없는 프롬프트로 다시 감싼 사본을 만든다."""
    rec = dict(base_record)
    rec["prompt"] = PROMPT_IMAGE_ONLY
    return rec


def augment_slang(records, include_text, mask_names):
    """FN 대응: 혐오 문장 속 원형 단어를 은어/변형어로 치환한 사본 추가."""
    extra = []
    for r in records:
        if r["label"] != "hate":
            continue
        text = r["source_text_raw"]
        n_added = 0
        for canonical, variants in SLANG_MAP.items():
            if canonical not in text:
                continue
            for variant in variants:
                if n_added >= MAX_SLANG_AUG_PER_RECORD:
                    break
                new_text = text.replace(canonical, variant)
                if new_text == text:
                    continue
                extra.append(make_record(
                    f"{r['id']}_slang_{variant}", r["image_path"], "hate", new_text,
                    include_text, mask_names, hate_categories=r.get("hate_categories", ""),
                    image_source=r.get("image_source", "unknown"),
                    # 은어 치환은 같은 이미지+같은 원본 텍스트의 변형이므로 두 그룹 다
                    # 원본 표본의 것을 그대로 물려받는다(별개 클러스터를 만들지 않음).
                    image_group_id=r["image_group_id"], text_group_id=r["text_group_id"],
                ))
                n_added += 1
            if n_added >= MAX_SLANG_AUG_PER_RECORD:
                break
    return extra


def augment_case_pairs(records, rng, include_text, mask_names, case_name,
                        image_from_label, text_from_label):
    """매칭된 케이스 생성: image_from_label 행의 이미지와 text_from_label 행의 텍스트를
    서로 다른 행끼리 크로스 페어링해서, 원본 metadata.csv엔 없는 조합을 만든다.

    조현서 케이스(image_hate_text_safe)는 image_from_label="hate", text_from_label="non_hate"
    (hate 이미지 x non_hate 텍스트). 정미정 케이스(text_hate_image_safe)는 그 반대
    (image_from_label="non_hate", text_from_label="hate": non_hate 이미지 x hate 텍스트).
    두 방향 모두 같은 함수로 만들어서 로직이 어긋나지 않게 한다.

    FN/FP augmentation(augment_slang/augment_counterfactual)과 달리 이건 실제 평가
    대상 케이스 자체를 만드는 것이므로, train/test 양쪽 split 모두에 적용해야
    evaluate.py --eval-filter로 이 케이스만 따로 채점할 수 있다. 같은 split 안에 있는
    행들끼리만 페어링하므로(convert_split이 split별로 호출) train/test 사이에
    이미지·텍스트가 섞이는 누수는 생기지 않는다.

    매칭 설계(analysis/matched_design_analysis.py)를 위해 image_group_id에는 이미지가
    가져온 원본 표본의 image_group_id를, text_group_id에는 텍스트가 가져온 원본 표본의
    text_group_id를 각각 그대로 물려준다(하나의 문자열로 합치지 않음 -- 합치면 조건마다
    합친 문자열이 서로 달라져서 "같은 이미지가 다른 조건에도 등장"하는 관계를 추적할 수
    없게 된다). 예를 들어 image_hate_text_safe 표본의 image_group_id는 그 이미지를 쓴
    자연 hate 표본의 image_group_id와 동일해서, assign_match_clusters()가 이 둘을 같은
    클러스터로 묶는다.
    """
    images = [
        (r["image_path"], r["image_group_id"], r.get("image_source", "unknown"))
        for r in records if r["label"] == image_from_label
    ]
    texts = [
        (r["id"], r["source_text_raw"], r["text_group_id"]) for r in records
        if r["label"] == text_from_label and r["source_text_raw"]
    ]
    if not images or not texts:
        return []

    images = images[:]
    texts = texts[:]
    rng.shuffle(images)
    rng.shuffle(texts)

    n_pairs = min(len(images), len(texts))
    extra = []
    for i in range(n_pairs):
        img_path, img_group, img_source = images[i]
        txt_id, text, txt_group = texts[i]
        new_id = f"{case_name}_{i:04d}_{txt_id}"
        extra.append(make_record(
            new_id, img_path, case_name, text, include_text, mask_names,
            hate_categories="", image_source=img_source,
            image_group_id=img_group, text_group_id=txt_group,
        ))
    return extra


def augment_counterfactual(records, rng, include_text, mask_names):
    """FP 대응: 민감 주제어 포함 비혐오 문장을 기존 non_hate 이미지와 짝지어 추가."""
    non_hate_by_path = {
        r["image_path"]: r["image_group_id"] for r in records if r["label"] == "non_hate"
    }
    if not non_hate_by_path:
        return []
    non_hate_images = list(non_hate_by_path.keys())
    extra = []
    i = 0
    for keyword, sentences in COUNTERFACTUAL_SENTENCES.items():
        for sentence in sentences:
            image_path = rng.choice(non_hate_images)
            new_id = f"cf_{keyword}_{i}"
            extra.append(make_record(
                new_id, image_path, "non_hate", sentence, include_text, mask_names,
                image_source="counterfactual_synthetic",
                # 재사용한 non_hate 이미지 쪽은 그 원본 표본과 같은 클러스터로 묶어야
                # 같은 이미지가 다른 표본에도 쓰였다는 사실 자체는 유지된다. 텍스트는
                # 완전히 새로 만든 문장이라 매칭 관계가 없으므로 새 id를 그대로 쓴다.
                image_group_id=non_hate_by_path[image_path],
                text_group_id=new_id,
            ))
            i += 1
    return extra


def write_jsonl(records, out_path):
    with open(out_path, "w", encoding="utf-8") as f_out:
        for r in records:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")


def report_label_balance(records, split):
    counts = Counter(r["label"] for r in records)
    total = sum(counts.values())
    parts = ", ".join(f"{k}={v} ({v/total:.1%})" for k, v in sorted(counts.items()))
    print(f"[{split}] label balance: {parts}")


def report_group_overlap(train_records, test_records):
    """중복 문장/유사 표본이 match_cluster_id로 묶여 있다면, train/test에 같은
    match_cluster_id가 걸쳐 있지 않은지 확인한다(데이터 분할 원칙: 동일 클러스터는
    같은 split에 배치). train/test는 원본 이미지 폴더 자체가 분리돼 있어 원래도 겹칠
    일이 없지만, 크로스 페어링이 두 split에서 독립적으로 수행됐는지 재확인하는
    용도다."""
    train_groups = {r["match_cluster_id"] for r in train_records}
    test_groups = {r["match_cluster_id"] for r in test_records}
    overlap = train_groups & test_groups
    if overlap:
        print(f"[경고] train/test에 걸쳐 있는 match_cluster_id {len(overlap)}개 발견 -- 데이터 "
              f"분할 원칙(동일 클러스터는 같은 split) 위반 가능성. 예: {list(overlap)[:5]}")
    else:
        print("[OK] train/test 간 match_cluster_id 중복 없음 (split 누수 없음).")


def report_token_length(records, out_dir, processor_name="NCSOFT/VARCO-VISION-2.0-1.7B"):
    """전체 문장의 token 길이 분포와, VARCO tokenizer 기준 과도하게 긴 표본 비율을
    학습 전에 출력한다. KELIP처럼 고정된 context_length=77 truncation은 없지만
    (VARCO는 causal LM이라 하드 컷오프가 없음), 길이가 길수록 이미지 타일 토큰과
    합쳐 OOM 위험이 커지므로, KELIP과 동일한 77-token 기준선으로 참고 비교하고
    그 이상인 표본은 원문·길이·라벨과 함께 별도 목록으로 저장해 검토한다."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(processor_name)
    except Exception as e:
        print(f"[경고] tokenizer를 불러오지 못해 token 길이 분포를 건너뜁니다 ({e}).")
        return

    lengths_by_record = [(r, len(tok.encode(r["source_text"]))) for r in records if r["source_text"]]
    if not lengths_by_record:
        print("[token 길이] source_text가 있는 표본이 없어 건너뜁니다.")
        return
    lengths = sorted(l for _, l in lengths_by_record)
    n = len(lengths)

    def pct(p):
        return lengths[min(n - 1, int(n * p))]

    threshold = 77  # KELIP과 동일 기준선으로 비교 참고용 (VARCO 자체는 하드 truncation 없음)
    over_records = [(r, l) for r, l in lengths_by_record if l > threshold]
    n_over = len(over_records)
    print(f"[token 길이] n={n} min={lengths[0]} p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} "
          f"max={lengths[-1]} | {threshold}-token 초과 표본 {n_over}개 ({n_over/n:.1%})")

    if n_over:
        out_path = os.path.join(out_dir, "long_text_samples.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r, l in over_records:
                f.write(json.dumps({
                    "id": r["id"], "label": r["label"], "token_length": l,
                    "truncated_at": threshold, "source_text": r["source_text"],
                }, ensure_ascii=False) + "\n")
        over_labels = Counter(r["label"] for r, _ in over_records)
        print(f"[token 길이] {threshold}-token 초과 표본을 {out_path}에 저장 "
              f"(라벨 분포: {dict(over_labels)})")


def convert_split(dataset_root, split, out_dir, include_text, mask_names, augment, case_pairs, rng):
    rows = read_rows(dataset_root, split)
    records = []
    n_skipped = 0
    for row in rows:
        label = row["label"].strip()
        if label not in ("hate", "non_hate"):
            continue
        image_path = row["image_path"]
        full_path = os.path.join(dataset_root, image_path)
        if not os.path.exists(full_path):
            n_skipped += 1
            print(f"[경고] 이미지 파일을 찾을 수 없어 표본 {row['id']}를 건너뜁니다: {full_path}")
            continue
        records.append(make_record(
            row["id"], image_path, label, row.get("source_text", "") or "",
            include_text, mask_names,
            hate_categories=row.get("hate_categories", "") or "",
            image_source=row.get("image_source", "unknown") or "unknown",
        ))
    if n_skipped:
        print(f"[{split}] 손상/누락 파일로 제외된 표본: {n_skipped}개")

    report_label_balance(records, split)

    n_base = len(records)
    extras = []
    n_image_hate_text_safe = n_text_hate_image_safe = 0
    if case_pairs:
        # train/test 양쪽 다: 두 케이스 다 augmentation이 아니라 평가도 필요한 정식
        # 케이스라서 test에도 있어야 --eval-filter로 각자 채점 가능.
        image_case = augment_case_pairs(
            records, rng, include_text, mask_names,
            case_name="image_hate_text_safe", image_from_label="hate", text_from_label="non_hate",
        )
        text_case = augment_case_pairs(
            records, rng, include_text, mask_names,
            case_name="text_hate_image_safe", image_from_label="non_hate", text_from_label="hate",
        )
        n_image_hate_text_safe, n_text_hate_image_safe = len(image_case), len(text_case)
        extras += image_case + text_case

    n_slang = n_cf = 0
    if augment and split == "train":
        slang_extra = augment_slang(records, include_text, mask_names)
        cf_extra = augment_counterfactual(records, rng, include_text, mask_names)
        n_slang, n_cf = len(slang_extra), len(cf_extra)
        extras += slang_extra + cf_extra

    records = records + extras
    if extras:
        rng.shuffle(records)

    # match_cluster_id는 반드시 이 split의 최종 표본 전체(자연 표본 + case-pairs +
    # slang/counterfactual augmentation)를 다 모은 뒤 한 번에 계산해야 한다 -- 부분
    # 집합에서 계산하면 나중에 추가되는 표본과의 연결(예: image_hate_text_safe가 참조하는
    # hate 이미지)이 반영되지 않는다.
    assign_match_clusters(records)

    out_path = os.path.join(out_dir, f"{split}.jsonl")
    write_jsonl(records, out_path)
    return records, n_base, n_slang, n_cf, n_image_hate_text_safe, n_text_hate_image_safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument(
        "--include-text",
        action="store_true",
        help="Include the paired source_text in the prompt (multimodal). "
        "Default is image-only, matching the 'image-part model' request.",
    )
    ap.add_argument(
        "--augment",
        action="store_true",
        help="Add slang/변형어 (FN) and counterfactual sensitive-keyword (FP) "
        "examples to the train split. Implies --include-text.",
    )
    ap.add_argument(
        "--case-pairs",
        action="store_true",
        help="박나현/조현서/정미정 통합 매칭 설계에 필요한 두 케이스를 모두 생성합니다: "
        "image_hate_text_safe(hate 이미지 x non_hate 텍스트, 조현서 케이스)와 "
        "text_hate_image_safe(non_hate 이미지 x hate 텍스트, 정미정 케이스). "
        "--augment와 달리 train/test 양쪽에 다 적용됩니다(평가에도 필요하므로). "
        "Implies --include-text.",
    )
    ap.add_argument(
        "--mask-person",
        action="store_true",
        help="model_text(=source_text/prompt)에도 실명을 <PERSON>으로 마스킹합니다. "
        "규칙이 팀 회의에서 확정되기 전까지는 기본 꺼짐(원문 유지)이며, source_text_raw는 "
        "이 옵션과 무관하게 항상 원문을 보존합니다.",
    )
    ap.add_argument(
        "--emit-modality-variants",
        action="store_true",
        help="Tier 2 라우팅(설계서 5장)에 필요한 image-only 프롬프트 변형을 "
        "{split}_image_only.jsonl로 추가 저장합니다.",
    )
    ap.add_argument(
        "--skip-token-report", action="store_true",
        help="VARCO tokenizer 로딩이 느리거나 불가능한 환경에서 token 길이 분포 출력을 건너뜁니다.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    include_text = args.include_text or args.augment or args.case_pairs
    if (args.augment or args.case_pairs) and not args.include_text:
        print("--augment/--case-pairs implies --include-text; enabling it.")

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    split_records = {}
    for split in ("train", "test"):
        records, n_base, n_slang, n_cf, n_iht, n_thi = convert_split(
            args.dataset_root, split, args.out_dir, include_text, args.mask_person,
            args.augment, args.case_pairs, rng,
        )
        split_records[split] = records
        total = n_base + n_slang + n_cf + n_iht + n_thi
        extra_bits = []
        if n_iht or n_thi:
            extra_bits.append(f"+{n_iht} image_hate_text_safe, +{n_thi} text_hate_image_safe")
        if n_slang or n_cf:
            extra_bits.append(f"+{n_slang} slang FN-aug, +{n_cf} counterfactual FP-aug")
        extra_msg = f" ({', '.join(extra_bits)})" if extra_bits else ""
        print(f"{split}: wrote {total} examples{extra_msg} -> {os.path.join(args.out_dir, split + '.jsonl')}")

        if args.emit_modality_variants:
            variant_path = os.path.join(args.out_dir, f"{split}_image_only.jsonl")
            write_jsonl([make_image_only_record(r) for r in records], variant_path)
            print(f"{split}: image-only 변형 {len(records)}개 -> {variant_path}")

    report_group_overlap(split_records["train"], split_records["test"])

    if include_text and not args.skip_token_report:
        all_records = split_records["train"] + split_records["test"]
        report_token_length(all_records, args.out_dir)


if __name__ == "__main__":
    main()
