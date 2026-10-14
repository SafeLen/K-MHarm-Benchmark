# 4.3 교차 결합의 자연도

조건 10·01은 서로 다른 게시물의 이미지와 텍스트를 무작위로 결합해 만든다. 이 결합의 어색함이 결과를 설명하는지 보기 위해, 결합의 자연도를 점수로 매기고 자연도를 반영해 다시 결합·학습·평가한다.

> **재구현 안내.** 논문 실험에 쓴 원래 코드는 보존되지 않아, 실험 기록(방법론 3.7)에 적힌 절차대로 다시 구현했다. 절차와 설정은 같지만 다음은 원래 값과 다를 수 있으므로, 다시 실행한 수치가 논문(R→R 81.1%, R→C 67.2%, C→C 71.5% 등)과 정확히 같지는 않다.
> - 자연도 질문 프롬프트의 문구 (`naturalness_lib.NAT_PROMPT`)
> - 후보 텍스트 40개와 seed별 20개를 고르는 난수
> - GPU 연산의 비결정성

## 절차

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 자연도 점수 | `naturalness_lib.py`, `score_naturalness.py` | 학습한 VARCO 모델 자체에 이미지와 글이 한 게시물로 함께 올라올 법한지 묻고, `log p('자연') − log p('어색')`(첫 토큰, teacher forcing)을 점수로 쓴다. 각 seed 모델이 자신의 평가셋 600건을 채점한다. |
| 판정자 타당성 | `naturalness_report.py validity` | 실제 게시물(조건 11·비혐오 300건)과 무작위 결합(조건 10·01 300건)을 점수로 구분하는 AUROC. 사전 기준은 0.65. |
| 자연도별 층화 | `naturalness_report.py stratify` | 조건 10·01에서 가장 어색한 50건과 가장 자연스러운 50건의 판정 차이(Multimodal 탐지율, Image Only·Text Only 혐오 판정 비율). seed 안 복원추출 부트스트랩 5,000회로 95% 신뢰구간. |
| 후보 채점 | `score_candidates.py` | 같은 split 안에서 이미지마다 후보 텍스트 40개를 뽑아 seed 1 모델로 채점한다. |
| 자연도 반영 결합 | `build_compat_pairs.py` | seed마다 후보 중 20개를 다시 골라, 각 이미지와 텍스트가 한 번씩 쓰이도록 점수 합이 최대인 선형 할당으로 짝을 만든다. 후보 밖 짝이 생기면 `pairing_summary.json`의 `n_fallback`에 남는다. |
| 학습·평가 (C→C) | `run_naturalness.sh` | 본 실험과 같은 설정(focal loss, modality dropout, 클래스 가중치, 검증 BA 임계값)으로 seed 1–5 학습·평가. |
| 교차 채점 (R→C) | `run_naturalness.sh` | 무작위 결합 모델을 같은 seed의 자연도 반영 평가셋에서 원래 임계값으로 채점. |
| 비교 | `naturalness_report.py compare`, `matched_design_analysis.py` | 세 구성(R→R, R→C, C→C)의 조건별 탐지율·FPR·Image Only 정확도와 조건 간 pairwise 대비(양방향 클러스터 부트스트랩). |

## 실행

`main_experiment`의 VARCO 본 실험(무작위 결합 R)을 먼저 끝내야 한다. `main_experiment/analysis/`의 `data_v2/`, `outputs_v2/seed<N>/lora-adapter/`, `outputs_v2_bal/seed<N>/`를 입력으로 쓴다.

```bash
cd naturalness
pip install -r requirements.txt
./run_naturalness.sh "1 2 3 4 5"                        # 전체
STEPS="nat report" ./run_naturalness.sh "1 2 3 4 5"     # 일부 단계만
```

결과는 `results/`(validity, stratify, compare 표와 pairwise 대비 보고서)에 저장된다. A100 기준으로 후보 채점(40,000쌍)이 가장 오래 걸린다.

## 해석상 주의

- 자연도 점수는 실제 혐오 게시물에서 실제 비혐오 게시물보다 낮게 나와 혐오 내용과 얽혀 있다. 층화 결과의 차이를 결합의 어색함만으로 설명할 수 없다.
- 자연도 반영 결합의 짝은 seed 1 모델의 점수로 모든 seed에 만든다.
