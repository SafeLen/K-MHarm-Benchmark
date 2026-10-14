# 이미지 데이터셋 재구성

## 분석 결론

이 폴더의 `metadata.csv`와 이미지 1,000개를 대조한 결과, 이미지는 게시글을 화면에
렌더링한 것이 아니라 `source_text`의 의미를 현실적인 한국 장면으로 바꾼 생성형
이미지입니다. `label`과 `hate_categories`는 저장·분류용 메타데이터이며 이미지 생성
입력으로 사용하지 않는 것이 데이터 누수를 막는 올바른 재구성입니다.

확정할 수 있는 사실:

- train 700개(혐오 350/비혐오 350), test 300개(혐오 150/비혐오 150)
- 원본은 모두 RGB `PNG`, `1024x1024`
- `image_dataset_small`은 동일 이미지의 `448x448` 변환본
- 소형본은 확장자가 `.png`지만 실제 파일 형식은 JPEG
- 원본을 Pillow `LANCZOS`로 448px로 줄인 후 JPEG `quality=90`,
  `optimize=False`로 저장하면 확인 표본이 아니라 전체 변환 규칙과 일치하며, 대표
  표본은 바이트 단위로 동일하게 재현됨

원본 파일에는 생성 모델명·프롬프트·seed 메타데이터가 없어 생성 단계의 정확한
프롬프트와 난수 결과까지 복원할 수는 없습니다. 시각적 특성, 1024px PNG 출력,
한국어 의미 해석과 장면 구성으로 보아 GPT Image 계열의 단일 이미지 생성 흐름이
가장 유력합니다. 코드는 당시 결과 재현 후보로 `gpt-image-1`, `medium`,
`1024x1024`를 기본값으로 둡니다. 이미지 생성은 확률적이므로 같은 입력으로 기존
픽셀과 동일한 그림이 나오지는 않습니다.

## 설치

```powershell
cd data_construction   # metadata.csv는 기본으로 ..\image_dataset 에서 읽습니다 (--dataset-root로 변경)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-generation.txt
$env:OPENAI_API_KEY = "본인의_API_키"
```

API 키는 코드나 CSV에 저장하지 않습니다. 현재 OpenAI 공식 문서는 단일 프롬프트의
이미지 생성에는 Image API를 사용하고, `size`, `quality`, `output_format`을 명시할
수 있다고 안내합니다.

## 안전하게 시험 실행

기존 이미지를 덮어쓰지 않도록 별도 출력 폴더를 사용합니다.

```powershell
python build_image_dataset.py generate `
  --output-root ..\image_dataset_regenerated `
  --limit 3 `
  --dry-run

python build_image_dataset.py generate `
  --output-root ..\image_dataset_regenerated `
  --limit 3 `
  --workers 1
```

`--dry-run`은 API를 호출하지 않습니다. 실제 생성 때는 완료·실패·프롬프트 해시·출력
해시가 `generation_manifest.jsonl`에 즉시 누적됩니다. 이미 존재하는 파일은 기본적으로
건너뛰므로 중단 후 같은 명령으로 재개할 수 있습니다. 다시 만들려면 명시적으로
`--overwrite`를 사용합니다.

현재 권장 모델로 실행하려면 모델만 교체할 수 있습니다.

```powershell
python build_image_dataset.py generate `
  --output-root ..\image_dataset_regenerated `
  --model gpt-image-2.5-flare `
  --quality medium `
  --workers 2
```

원본과 최대한 같은 계열을 추정 재현하는 목적이면 기본 `gpt-image-1`을 유지합니다.
해당 구형 모델을 계정에서 더 이상 사용할 수 없을 때 위와 같이 현재 모델을 지정합니다.

## 448px 소형본의 정확한 변환

아래 명령은 API를 사용하지 않습니다. 출력 파일은 역사적 데이터와 맞추기 위해
JPEG 바이트를 `.png` 파일명으로 저장합니다.

```powershell
python build_image_dataset.py resize `
  --source-root ..\image_dataset_1024 `
  --output-root ..\image_dataset_small_rebuilt
```

새 파이프라인에서는 확장자와 실제 포맷을 일치시키는 편이 낫지만, 기존 학습 코드 및
소형본과의 완전한 호환을 위해 재현 명령은 원래 동작을 보존했습니다. Pillow는 파일
내용을 감지하므로 기존 학습 로더에서는 정상적으로 열립니다.

## 검증

```powershell
python build_image_dataset.py audit --image-root ..\image_dataset_1024
python build_image_dataset.py audit --image-root ..\image_dataset_small_rebuilt
```

검증은 CSV 스키마, ID 중복, 경로 이탈, split/label 경로 일치, 누락·손상 이미지,
이미지 형식과 크기를 검사합니다. `--splits train`, `--ids ID1 ID2`, `--limit N`으로
대상을 좁힐 수 있습니다.

## 전체 제작 흐름

1. 각 split의 `metadata.csv`를 UTF-8 CSV로 읽고 필수 열과 경로를 검증합니다.
2. 오직 `source_text`만 프롬프트 템플릿에 넣습니다. CSV의 라벨은 출력 경로 결정에만
   쓰고 이미지 모델에는 보내지 않습니다.
3. 1024x1024 불투명 PNG 한 장을 생성하고 디코딩·크기를 검증합니다.
4. 임시 파일에 먼저 쓴 뒤 원자적으로 최종 경로로 이동하여 중간에 끊긴 파일을 막습니다.
5. 재시도 가능한 429/5xx 오류는 지수 백오프로 재시도하고, 모든 결과를 manifest에
   기록합니다.
6. 필요하면 확정된 LANCZOS/JPEG 변환으로 448px 소형본을 만듭니다.

생성 이미지에는 원문의 유해 표현이 시각적으로 과장되거나 보호 집단의 고정관념으로
바뀌지 않도록 중립적·비그래픽 묘사 지침을 넣었습니다. 일부 입력은 서비스의 안전
정책에 따라 거절될 수 있으며, 코드는 이를 숨기거나 우회하지 않고 실패로 기록합니다.
