# 한국어 멀티모달 유해성 탐지 벤치마크

**한국어 멀티모달 유해성 탐지 벤치마크 및 비전-언어 모델의 모달리티 의존성 분석**
(A Korean Multimodal Harmfulness Detection Benchmark and an Analysis of Modality Dependence in Vision-Language Models)

유해 정보가 이미지와 텍스트 양쪽(11), 이미지만(10), 텍스트만(01), 어느 쪽에도 없는(00) 네 조건으로 구성한 한국어 멀티모달 벤치마크와,
같은 게시물 정답으로 Multimodal(MM)·Text Only(TO)·Image Only(IO) 입력을 비교하는 평가 코드입니다.

## 폴더와 논문 절의 대응

| 폴더 | 내용 | 논문 |
|---|---|---|
| `image_dataset/` | 데이터셋: 원문 1,000개와 이미지 1,000장(448px), `{train,test}/metadata.csv` | 3.1, 3.2 |
| `data_construction/` | UnSmile 라벨 재정의·원문 1,000개 추출(`build_unsmile_subset.py`), gpt-image-1 이미지 생성(`build_image_dataset.py`) | 3.1, 3.2 |
| `main_experiment/shared/` | VARCO 본 실험: 교차 결합 데이터 생성, LoRA 학습(focal loss, label smoothing, modality dropout, 클래스 가중치), 평가 | 3.3, 3.4 |
| `main_experiment/analysis/` | 5 seed 실행(`run_multi_seed.sh`), BA 임계값 재선택, 혼합효과 회귀·클러스터 부트스트랩(`matched_design_analysis.py`) | 3.5, 4.2 표 4 |
| `vlm_compare/` | 5개 VLM 미세조정 비교(`train_lora.py`, `evaluate.py`, `mask_eval.py`)와 7개 VLM Zero-shot(`zeroshot_score.py`), 공통 split(`data/`) | 3.3, 표 3, 부록 A |
| `text_baseline/` | 텍스트 전용 KcELECTRA baseline | 3.3, 표 3 |
| `evaluation/` | 공통 평가(`eval_common.py`), 결과 형식 변환, 게시물 단위 McNemar+BH(`mcnemar_post.py`), 그림 2 | 3.5, 표 3, 4.2, 그림 2 |
| `preds/` | 표 3에 쓴 모든 예측 점수 (공통 형식) | 표 3 |
| `results/` | `eval_common.py`·`mcnemar_post.py` 출력과 그림 2 | 표 3, 4.2, 그림 2 |
| `naturalness/` | 4.3 교차 결합의 자연도 (TODO) | 4.3 |

## 데이터셋

| | train | test |
|---|---|---|
| 혐오 원문·이미지 (`hate/`) | 350 | 150 |
| 무해 원문·이미지 (`non_hate/`) | 350 | 150 |
| 교차 결합 후 게시물 (조건 11·10·01·00 각 1/4) | 1,400 | 600 |

- `image_dataset/{train,test}/metadata.csv`: `id, label, source_text, image_path, hate_categories`
- 게시물(조건 11·10·01·00)은 코드가 같은 split 안에서 이미지와 텍스트를 교차 결합해 만듭니다 (`vlm_compare/data_prep.py`, `main_experiment/shared/data_prep.py`). 미세조정 비교와 Zero-shot에 쓴 split은 `vlm_compare/data/*.jsonl`에 고정해 두었습니다.
- **이미지 두 가지 크기**
  - 저장소의 `image_dataset/`: 448×448 축소본. 모든 실험은 이미지를 384px로 다시 줄여 쓰므로 실험 재현에는 이것으로 충분합니다. 파일 이름은 `.png`지만 실제 형식은 JPEG(quality 90)입니다. Pillow는 내용으로 형식을 판별하므로 그대로 열립니다.
  - 1024×1024 원본 PNG: [Releases](../../releases)의 `image_dataset_1024.zip`(약 1.4GB)입니다. 풀면 같은 폴더 구조가 나오며, `data_construction/build_image_dataset.py resize`로 저장소의 448px 파일을 바이트 단위까지 똑같이 다시 만들 수 있습니다.
- 이미지의 유해 여부는 원문 라벨을 상속한 값이며 독립적인 시각 주석으로 검증하지 않았습니다.

## 공통 형식과 평가 규칙

모든 모델의 예측은 `sample_id, condition(11/10/01/00), input_mode(MM/TO/IO), score` 열을 가진 CSV로 저장하고 `evaluation/eval_common.py`로 같은 정의에 따라 채점합니다.

- **정답은 게시물 단위입니다.** 조건 11·10·01은 혐오, 00은 비혐오이며, 입력을 가려도 정답은 바뀌지 않습니다.
- **점수**는 `log p(혐오|입력) − log p(비혐오|입력)`입니다.
- **임계값**은 검증셋에서 Balanced Accuracy가 최대인 값을 test에 고정합니다. 미세조정 VLM의 TO·IO에는 MM 검증 임계값을, Zero-shot에는 입력 방식별 검증 임계값을 씁니다.

## 표·그림 재현 (GPU 불필요)

`preds/`의 예측으로 표 3과 그림 2를 다시 만들 수 있습니다. 저장소 루트에서 실행합니다.

```bash
pip install -r evaluation/requirements.txt
python evaluation/eval_common.py --pred "preds/zeroshot_*.csv"        --out results/zeroshot
python evaluation/eval_common.py --pred "preds/modelcompare_*.csv"    --out results/model_compare
python evaluation/eval_common.py --pred "preds/varco5seed_*.csv"      --out results/varco5seed
python evaluation/eval_common.py --pred "preds/text_kcelectra_seed*.csv" --out results/text_kcelectra
python evaluation/mcnemar_post.py --pred "preds/varco5seed_*.csv" --per-run results/varco5seed/per_run.csv --out results/varco5seed
python evaluation/make_fig2.py --summary results/varco5seed/summary.csv --out results/fig2_post_level.png
```

## 실험 전체 재현

### 1. 데이터

```bash
# UnSmile 공식 TSV(unsmile_train_v1.0.tsv, unsmile_valid_v1.0.tsv)를 data_construction/unsmile/에 받는다
#   https://github.com/smilegate-ai/korean_unsmile_dataset
cd data_construction
python build_unsmile_subset.py verify --unsmile-dir unsmile --dataset-root ../image_dataset   # 공개 데이터 점검
# 새로 만들 때: select -> build_image_dataset.py generate (OPENAI_API_KEY 필요, README_REBUILD.md 참고)
```

저장소의 `image_dataset/`가 이후 모든 단계의 입력이며, 아래 명령의 `--dataset-root`는 모두 이 폴더를 가리킵니다.
논문에 쓴 1,000개는 원래 추출에서 정해졌습니다. 그 id와 난수 상태가 남아 있지 않아 `select`로는 같은 표본이 다시 나오지 않으므로, 재현에는 저장소의 `image_dataset/`를 그대로 쓰세요.

### 2. VARCO 본 실험 (5 seed)

```bash
cd main_experiment/analysis
pip install -r ../shared/requirements.txt -r requirements.txt
DATASET_ROOT=../../image_dataset DATA_DIR=data_v2 OUT_ROOT=outputs_v2 ./run_multi_seed.sh "1 2 3 4 5"
# val BA 최대 임계값을 seed마다 threshold.json으로 저장하고 test 예측을 outputs_v2_bal/로 옮긴다
python reselect_threshold.py --dataset-root ../../image_dataset --data-dir data_v2 \
    --in-root outputs_v2 --out-root outputs_v2_bal --seeds 1,2,3,4,5 --strategy balanced_accuracy
python matched_design_analysis.py --outputs-root outputs_v2_bal --seeds 1,2,3,4,5 \
    --report-out outputs_v2_bal/matched_design_report.md                        # 표 4, 혼합효과 회귀
python ../../evaluation/convert_existing.py varco-seeds --root outputs_v2_bal --data-dir data_v2 --out ../../preds
```

### 3. 미세조정 비교 (seed 42, plain cross-entropy, modality dropout 없음)

```bash
cd vlm_compare
pip install -r requirements.txt
# data/{train,val,test}.jsonl 은 data_prep.py --seed 42 결과이며 Zero-shot과 같은 split이다
for m in varco hyperclovax bllossom ax; do
  python train_lora.py --model $m --dataset-root ../image_dataset
  python evaluate.py   --model $m --dataset-root ../image_dataset          # MM
  python mask_eval.py  --model $m --mask image --dataset-root ../image_dataset   # TO (회색 이미지)
  python mask_eval.py  --model $m --mask text  --dataset-root ../image_dataset   # IO (게시글 비움)
done
# qwen3vl은 transformers==4.57.1, peft==0.17.1 로 올린 뒤 같은 명령을 실행
python collect_results.py && python ablation_summary.py
python ../evaluation/convert_existing.py model-compare --root outputs --out ../preds
```

### 4. Zero-shot (7개 모델)

```bash
cd vlm_compare
for m in varco hyperclovax bllossom ax internvl3 llavaov; do
  python zeroshot_score.py --model $m --dataset-root ../image_dataset --val-jsonl data/val.jsonl
done
# qwen3vl은 transformers==4.57.1 환경에서 실행
python ../evaluation/convert_existing.py zeroshot --root results_zeroshot --out ../preds
```

### 5. 텍스트 전용 baseline

```bash
cd text_baseline
for s in 1 2 3; do python text_baseline.py --seed $s; done   # UnSmile에서 test 문장을 뺀 11,626문장으로 학습
```

## 참고 사항

- **Image Only 프롬프트**: Zero-shot은 게시글 줄을 뺀 프롬프트(부록 A)를, 미세조정 비교는 학습 때와 같은 프롬프트에 게시글을 빈 문자열로 넣었습니다. VARCO 본 실험은 `main_experiment`의 `test_image_only.jsonl`을 씁니다.
- **VARCO 본 실험의 비혐오 TO/IO 점수**: 원래 저장되지 않아, 같은 텍스트(조건 10 TO)·같은 이미지(조건 01 IO)의 점수를 복사했습니다. 같은 입력이면 점수가 같음을 확인했고, 복사한 행은 `derived_from` 열에 원본 id가 있습니다.
- **Zero-shot HyperCLOVAX**: 표 3은 다시 채점한 결과(`preds/zeroshot_hyperclovax.csv`)를 씁니다. bf16 연산 차이로 처음 채점한 결과와 점수가 조금 다릅니다.
- **LoRA 가중치와 학습 로그**는 용량 때문에 저장소에 넣지 않았습니다.

## 라이선스와 데이터 출처

- 텍스트: Korean UnSmile Dataset (Kang et al., 2022). 원 데이터의 라이선스를 따릅니다.
- 이미지: 각 원문을 입력으로 OpenAI gpt-image-1이 생성한 1024×1024 이미지입니다. 프롬프트에는 혐오 여부와 범주를 넣지 않았습니다.
- 이 데이터에는 혐오 표현이 포함되어 있습니다. 연구 목적으로만 사용하세요.
