#!/usr/bin/env bash
# 통합 matched-design 학습(모델 1개, 세 조건을 합친 데이터) x 5+ seed 오케스트레이션.
#
# GPU 환경(A100 80GB에서 실행)에서 shared/ 아래 스크립트를 이 순서대로 호출한다.
# seed 범위를 나눠 여러 머신에서 돌린 뒤 outputs/seed<N>/ 폴더를 한 곳에 모아
# analysis/matched_design_analysis.py로 합쳐서 분석해도 된다.
#
# 논문 결과(VARCO 본 실험)는 DATA_DIR=data_v2, OUT_ROOT=outputs_v2로 f1 임계값으로 학습한 뒤
# reselect_threshold.py로 val BA 최대 임계값을 다시 골라(outputs_v2_bal) 얻었다. 임계값은 학습이
# 끝난 뒤 고르므로, 아래처럼 처음부터 --threshold-strategy balanced_accuracy로 돌린 것과 같다.
#
# Usage:
#   ./run_multi_seed.sh "1 2 3 4 5"
set -euo pipefail

SEEDS="${1:-1 2 3 4 5}"
DATASET_ROOT="${DATASET_ROOT:-../image_dataset}"
SHARED_DIR="${SHARED_DIR:-../shared}"
DATA_DIR="${DATA_DIR:-data}"
OUT_ROOT="${OUT_ROOT:-outputs}"

# 매칭 설계(H_T/H_I)용 세 조건. matched_design_analysis.py의 CONDITIONS와 일치해야 한다.
CONDITIONS="hate image_hate_text_safe text_hate_image_safe"
# sanity-check용: 세 조건만 채점하면 recall(탐지율)만 보게 되어 "무조건 혐오"로 퇴화한
# 모델도 만점처럼 보일 수 있다. non_hate/all도 채점해서 전체 accuracy/FPR을 확인한다
# (matched_design_analysis.py의 리포트 0번 섹션이 이 값을 읽는다).
SANITY_FILTERS="non_hate all"

for SEED in $SEEDS; do
  echo "=================================================================="
  echo "seed=$SEED"
  echo "=================================================================="

  SEED_DATA_DIR="$DATA_DIR/seed$SEED"
  SEED_OUT_DIR="$OUT_ROOT/seed$SEED"
  ADAPTER_DIR="$SEED_OUT_DIR/lora-adapter"
  mkdir -p "$SEED_DATA_DIR" "$SEED_OUT_DIR"

  # 1) 데이터 준비: 세 조건(hate/image_hate_text_safe/text_hate_image_safe)을 모두
  #    포함한 하나의 train/test.jsonl + image_only 변형(모달리티 ablation용)을 생성.
  #    --seed는 크로스 페어링(augment_case_pairs)에 쓰는 rng seed -- 매칭 자체는 seed마다
  #    다르게 뽑히지만(재현성 확인용), 학습/평가 seed(아래 --seed)와는 별개 개념이다.
  # seed1에서 --augment + --threshold-strategy precision_constrained을 같이 시도해봤으나
  # (동점 처리 버그도 고친 뒤에도) non_hate FPR이 기본 f1 threshold보다 오히려 나빴다
  # (v1 f1: FPR 32.7% < v3 augment+precision_constrained: FPR 43.3%). validation set이
  # 작아서(n=143) precision_constrained이 과적합되듯 선택된 것으로 보임, --augment도
  # counterfactual 반례가 10개뿐이라 효과가 제한적이었다. 그래서 기본 f1 threshold로
  # 되돌리고(--augment도 뺌), non_hate FPR 자체는 5개 seed를 모아 matched_design_analysis.py
  # (섹션 0 sanity check)로 더 안정적으로 판단한다. select_threshold_on_val의 동점 처리
  # 버그 수정(train_lora.py)은 그대로 유지 -- precision_constrained을 나중에 다시 쓸 때
  # 필요한 정당한 수정이므로.
  python "$SHARED_DIR/data_prep.py" \
    --dataset-root "$DATASET_ROOT" \
    --out-dir "$SEED_DATA_DIR" \
    --include-text --case-pairs --emit-modality-variants \
    --seed "$SEED"

  # 2) 학습: 모델 1개만, 세 조건이 합쳐진 train.jsonl 전체로 학습(공유 checkpoint).
  #    modality-dropout/aux-loss/tune-vision 등은 필요에 따라 조정.
  #    --class-balance: 정답이 '혐오'인 표본이 non_hate의 약 3배라 혐오 쪽으로 치우쳐
  #    non_hate FPR이 높았다(seed1~4 실측 10.7~40.0%). 역빈도 loss 가중치로 보정한다.
  #    --threshold-strategy balanced_accuracy: val도 정답의 약 3/4이 혐오라 기본 f1 기준은
  #    임계값을 혐오 쪽으로 크게 낮춘다(v2 seed1: f1 -1.88 -> test FPR 34.7%). 이미 f1로
  #    학습한 seed는 reselect_threshold.py로 재학습 없이 임계값만 다시 고를 수 있다.
  #    (modality-dropout은 이제 이미지를 가린 표본의 정답을 텍스트 단독 정답으로 바꾼다 --
  #    train_lora.py HateVLMDataset 참고.)
  python "$SHARED_DIR/train_lora.py" \
    --dataset-root "$DATASET_ROOT" \
    --train-jsonl "$SEED_DATA_DIR/train.jsonl" \
    --output-dir "$ADAPTER_DIR" \
    --epochs 3 --batch-size 2 --grad-accum 16 \
    --focal-gamma 2.0 --label-smoothing 0.05 \
    --modality-dropout 0.2 --class-balance \
    --threshold-strategy balanced_accuracy \
    --val-frac 0.15 --early-stopping-patience 2 \
    --seed "$SEED"

  # 3) 평가: 조건별(hate/image_hate_text_safe/text_hate_image_safe) x
  #    모달리티별(full/text_only/image_only) 예측 csv를 matched_design_analysis.py가
  #    기대하는 파일명 규칙으로 저장.
  for COND in $CONDITIONS; do
    python "$SHARED_DIR/evaluate.py" \
      --dataset-root "$DATASET_ROOT" \
      --test-jsonl "$SEED_DATA_DIR/test.jsonl" \
      --adapter-dir "$ADAPTER_DIR" \
      --eval-filter "$COND" \
      --out "$SEED_OUT_DIR/test_predictions_${COND}_full.csv"

    python "$SHARED_DIR/evaluate.py" \
      --dataset-root "$DATASET_ROOT" \
      --test-jsonl "$SEED_DATA_DIR/test.jsonl" \
      --adapter-dir "$ADAPTER_DIR" \
      --eval-filter "$COND" --blank-image \
      --out "$SEED_OUT_DIR/test_predictions_${COND}_text_only.csv"

    python "$SHARED_DIR/evaluate.py" \
      --dataset-root "$DATASET_ROOT" \
      --test-jsonl "$SEED_DATA_DIR/test_image_only.jsonl" \
      --adapter-dir "$ADAPTER_DIR" \
      --eval-filter "$COND" \
      --out "$SEED_OUT_DIR/test_predictions_${COND}_image_only.csv"
  done

  # 3b) Sanity check: non_hate(오탐률)와 all(전체 accuracy/AUROC) -- full modality만.
  #     matched_design_analysis.py의 build_report() 0번 섹션이 이 두 파일을 읽는다.
  for COND in $SANITY_FILTERS; do
    python "$SHARED_DIR/evaluate.py" \
      --dataset-root "$DATASET_ROOT" \
      --test-jsonl "$SEED_DATA_DIR/test.jsonl" \
      --adapter-dir "$ADAPTER_DIR" \
      --eval-filter "$COND" \
      --out "$SEED_OUT_DIR/test_predictions_${COND}_full.csv"
  done

done

echo "모든 seed 완료. 다음으로 분석을 돌리세요:"
echo "  python ../analysis/matched_design_analysis.py --outputs-root $OUT_ROOT --seeds ${SEEDS// /,}"
