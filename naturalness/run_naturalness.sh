#!/usr/bin/env bash
# 4.3 교차 결합의 자연도 (방법론 3.7) 전체 실행. naturalness/ 에서 실행한다.
# main_experiment/analysis/README의 순서로 VARCO 본 실험(무작위 결합 R)을 먼저 끝내야 한다:
#   data_v2/seed<N>/, outputs_v2/seed<N>/lora-adapter/, outputs_v2_bal/seed<N>/(threshold.json, 예측)
#
# Usage:  STEPS="nat report candidates pairs train_c cross analyze" ./run_naturalness.sh "1 2 3 4 5"
set -euo pipefail

SEEDS="${1:-1 2 3 4 5}"
STEPS="${STEPS:-nat report candidates pairs train_c cross analyze}"
DATASET_ROOT="${DATASET_ROOT:-../image_dataset}"
SHARED_DIR="${SHARED_DIR:-../main_experiment/shared}"
ANALYSIS_DIR="${ANALYSIS_DIR:-../main_experiment/analysis}"
R_DATA="${R_DATA:-$ANALYSIS_DIR/data_v2}"            # 무작위 결합 데이터
R_ADAPTERS="${R_ADAPTERS:-$ANALYSIS_DIR/outputs_v2}"  # 무작위 결합 모델
R_PRED="${R_PRED:-$ANALYSIS_DIR/outputs_v2_bal}"     # 무작위 결합 모델의 BA 임계값과 예측 (R->R)
CANDIDATE_SEED_MODEL="${CANDIDATE_SEED_MODEL:-1}"    # 후보 채점 모델 (논문: seed 1)

CONDITIONS="hate image_hate_text_safe text_hate_image_safe"
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

# evaluate.py로 조건 x 입력 방식 예측을 run_multi_seed.sh와 같은 파일명으로 저장한다.
# $1 adapter  $2 data dir (test.jsonl, test_image_only.jsonl)  $3 out dir  $4.. evaluate.py 추가 인자
eval_all() {
  local adapter="$1" data="$2" out="$3"; shift 3
  mkdir -p "$out"
  for COND in $CONDITIONS; do
    python "$SHARED_DIR/evaluate.py" --dataset-root "$DATASET_ROOT" --test-jsonl "$data/test.jsonl" \
      --adapter-dir "$adapter" --eval-filter "$COND" --out "$out/test_predictions_${COND}_full.csv" "$@"
    python "$SHARED_DIR/evaluate.py" --dataset-root "$DATASET_ROOT" --test-jsonl "$data/test.jsonl" \
      --adapter-dir "$adapter" --eval-filter "$COND" --blank-image \
      --out "$out/test_predictions_${COND}_text_only.csv" "$@"
    python "$SHARED_DIR/evaluate.py" --dataset-root "$DATASET_ROOT" --test-jsonl "$data/test_image_only.jsonl" \
      --adapter-dir "$adapter" --eval-filter "$COND" --out "$out/test_predictions_${COND}_image_only.csv" "$@"
  done
  for COND in non_hate all; do
    python "$SHARED_DIR/evaluate.py" --dataset-root "$DATASET_ROOT" --test-jsonl "$data/test.jsonl" \
      --adapter-dir "$adapter" --eval-filter "$COND" --out "$out/test_predictions_${COND}_full.csv" "$@"
  done
}

# 3.7.2 각 seed 모델이 자신의 평가셋 600건에 자연도 점수를 매긴다
if has_step nat; then
  for S in $SEEDS; do
    python score_naturalness.py --adapter-dir "$R_ADAPTERS/seed$S/lora-adapter" \
      --jsonl "$R_DATA/seed$S/test.jsonl" --dataset-root "$DATASET_ROOT" --out "nat/nat_seed$S.csv"
  done
fi

# 3.7.3 판정자 타당성 + 자연도별 층화 (표 10, 표 11)
if has_step report; then
  python naturalness_report.py validity --nat-dir nat --seeds $SEEDS --out results
  python naturalness_report.py stratify --nat-dir nat --pred-root "$R_PRED" --seeds $SEEDS --out results
fi

# 3.7.4 후보 40개 채점 (seed 1 모델) -> seed마다 20개 선택 + 선형 할당
if has_step candidates; then
  python score_candidates.py --adapter-dir "$R_ADAPTERS/seed$CANDIDATE_SEED_MODEL/lora-adapter" \
    --dataset-root "$DATASET_ROOT" --out-dir candidates
fi
if has_step pairs; then
  python build_compat_pairs.py --dataset-root "$DATASET_ROOT" --candidates-dir candidates \
    --out-dir data_compat --seeds $SEEDS
fi

# 3.7.4 자연도 반영 결합으로 본 실험과 같은 설정으로 학습·평가 (C->C)
if has_step train_c; then
  for S in $SEEDS; do
    python "$SHARED_DIR/train_lora.py" --dataset-root "$DATASET_ROOT" \
      --train-jsonl "data_compat/seed$S/train.jsonl" --output-dir "outputs_compat/seed$S/lora-adapter" \
      --epochs 3 --batch-size 2 --grad-accum 16 --focal-gamma 2.0 --label-smoothing 0.05 \
      --modality-dropout 0.2 --class-balance --threshold-strategy balanced_accuracy \
      --val-frac 0.15 --early-stopping-patience 2 --seed "$S"
    eval_all "outputs_compat/seed$S/lora-adapter" "data_compat/seed$S" "outputs_compat/seed$S"
  done
fi

# 3.7.5 교차 채점: 무작위 결합 모델을 같은 seed의 자연도 반영 평가셋에서, 원래 BA 임계값으로 채점 (R->C)
if has_step cross; then
  for S in $SEEDS; do
    THR=$(python -c "import json;print(json.load(open('$R_PRED/seed$S/threshold.json'))['new_threshold'])")
    eval_all "$R_ADAPTERS/seed$S/lora-adapter" "data_compat/seed$S" "outputs_r2c/seed$S" --threshold "$THR"
  done
fi

# 표 12 (세 구성 비교)와 표 13 (조건 간 pairwise 대비, 양방향 클러스터 부트스트랩)
if has_step analyze; then
  SEED_CSV=$(echo $SEEDS | tr ' ' ',')
  python naturalness_report.py compare --seeds $SEEDS --out results \
    --config "R->R=$R_PRED" --config "R->C=outputs_r2c" --config "C->C=outputs_compat"
  for OUT in outputs_r2c outputs_compat; do
    python "$ANALYSIS_DIR/matched_design_analysis.py" --outputs-root "$OUT" --seeds "$SEED_CSV" \
      --report-out "results/matched_design_report_${OUT#outputs_}.md"
  done
fi
