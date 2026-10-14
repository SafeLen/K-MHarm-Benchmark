"""
Matched-design 통계 분석: 박나현(11, hate) / 조현서(10, image_hate_text_safe) /
정미정(01, text_hate_image_safe) 세 조건에 대해, 사용자가 공유한 방법론(매칭
quadruple, 단일 checkpoint, mixed-effects 로지스틱 회귀, bootstrap 상호작용 CI,
5+ seed, McNemar)을 **00(무해 이미지+무해 텍스트) 조건이 없는 3-조건 버전으로 적용**한다.

=== 왜 00이 없고, 그게 분석에 어떤 의미인지 ===
이번 연구 설계는 00(무해 이미지+무해 텍스트) 조건을 포함하지 않기로 확정했다(사용자
확인). H_T, H_I를 각각 0/1 이진 변수로 두면:
    11 = (H_T=1, H_I=1)  박나현
    10 = (H_T=0, H_I=1)  조현서   (텍스트 무해, 이미지 유해)
    01 = (H_T=1, H_I=0)  정미정   (텍스트 유해, 이미지 무해)
    00 = (H_T=0, H_I=0)  -- 없음
절편 + H_T + H_I + H_T:H_I(상호작용) 모델은 파라미터 4개인데 관측된 셀(cell)이 3개뿐이라
**상호작용 항(H_T:H_I, 즉 Delta = mu11 - mu10 - mu01 + mu00)은 대수적으로 추정 불가능**
하다(rank-deficient). 그래서 이 스크립트는:
    1) 상호작용 항이 없는 주효과(H_T, H_I)만의 mixed-effects 로지스틱 회귀를 적합하고,
    2) 대신 11 vs 10, 11 vs 01, 10 vs 01의 pairwise 대비를 cluster bootstrap CI로 보고한다.
00 데이터가 나중에 추가되면 fit_mixed_effects의 formula에 상호작용 항을 더해 Delta를
직접 계산하도록 확장할 수 있다.

=== 매칭 구조: crossed(교차) random effects, 단일 group_id 아님 ===
data_prep.py의 augment_case_pairs()는 image_group_id(이 표본 이미지의 원본 출처)와
text_group_id(이 표본 텍스트의 원본 출처)를 **따로** 남긴다. 예를 들어 조현서 케이스
(image_hate_text_safe) 표본의 image_group_id는 그 이미지를 쓴 박나현 케이스(자연 hate)
표본의 image_group_id와 같다 -- 이게 "같은 이미지가 다른 조건에도 등장"하는 매칭
관계의 실체다. 이 둘을 하나의 문자열로 합치면(예: "casepair_img_txt") 조건마다 합친
문자열이 달라져 매칭이 전혀 추적되지 않으므로, 이 스크립트는 image_group_id와
text_group_id를 **교차(crossed) random effect 두 개**로 다룬다(하나의 그룹으로
합치지 않음).

주의: data_prep.py의 --case-pairs는 모든 hate/non_hate 표본의 이미지·텍스트를 한 번씩
전부 재사용해 크로스 페어링을 만들기 때문에, "완전 연결요소(connected component)"
기준으로 묶으면 표본 대부분이 하나의 거대 클러스터로 합쳐진다(수학적으로 불가피한
현상 -- train_lora.py의 split_train_val 참고). 그래서 여기서는 image/text를 별도
random effect로 다루는 crossed 모델을 쓰고, bootstrap도 image_group_id와
text_group_id를 **독립적으로** 리샘플하는 two-way cluster bootstrap을 쓴다(하나로
합쳐서 리샘플하면 같은 거대 클러스터 붕괴 문제가 생김).

=== 입력 ===
evaluate.py가 조건(--eval-filter)별 · 모달리티(full/text_only/image_only)별 · seed별로
저장한 예측 csv들. 파일명 규칙(run_multi_seed.sh가 이 규칙대로 저장함):
    outputs/seed<S>/test_predictions_<condition>_full.csv
    outputs/seed<S>/test_predictions_<condition>_text_only.csv   (evaluate.py --blank-image)
    outputs/seed<S>/test_predictions_<condition>_image_only.csv  (*_image_only.jsonl로 채점)
각 csv는 evaluate.py가 쓰는 컬럼(id, image_group_id, text_group_id, match_cluster_id,
true_label, true_target, pred_label, score, prob_hate)을 갖는다.

또한 non_hate/all(전체 test set) 채점 결과도 별도 섹션(섹션 0)에서 함께 읽어 overall
accuracy/FPR을 보고한다 -- 세 혐오 조건(hate/image_hate_text_safe/text_hate_image_safe)
만 --eval-filter로 채점하면 recall(탐지율)만 보게 되어, "무조건 혐오라고 답하는" 퇴화
모델도 만점처럼 보일 수 있다. non_hate/all 채점이 있어야 오탐률(FPR)과 진짜 accuracy를
확인할 수 있다.

Usage:
    python matched_design_analysis.py --outputs-root outputs --seeds 1,2,3,4,5 \
        --report-out outputs/matched_design_report.md
"""
import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

CONDITIONS = {
    # condition_label -> (H_T, H_I)  (팀 정의: 1=유해, 0=무해)
    "hate": (1, 1),                    # 박나현
    "image_hate_text_safe": (0, 1),    # 조현서
    "text_hate_image_safe": (1, 0),    # 정미정
}
CONDITION_OWNER = {
    "hate": "박나현",
    "image_hate_text_safe": "조현서",
    "text_hate_image_safe": "정미정",
}
MODALITIES = ("full", "text_only", "image_only")


def expected_hate(modality, h_t, h_i):
    """모달리티별 "모델이 그 입력만 봤을 때의 정답"(1=혐오). full은 팀 OR 규칙, text_only는
    텍스트 유해 여부(H_T), image_only는 이미지 유해 여부(H_I). train_lora.py의
    CASE_MODALITY_TARGETS와 같은 정의다.

    이전 버전은 모든 모달리티에서 correct = (pred == 혐오)로 계산해서, 예컨대
    text_hate_image_safe(01)의 image_only(무해 이미지만 보여줌)에서 "비혐오"라고 맞게
    답한 것을 오답으로 셌고, McNemar(full vs text_only/image_only) 비교의 "정답" 의미가
    모달리티마다 섞여 있었다."""
    if modality == "full":
        return int(bool(h_t or h_i))
    if modality == "text_only":
        return int(h_t)
    if modality == "image_only":
        return int(h_i)
    raise ValueError(f"알 수 없는 modality: {modality}")
# 매칭 설계(H_T/H_I)에는 안 쓰지만, "무조건 혐오"로 퇴화한 모델을 잡아내기 위한
# 전체 정확도/FPR sanity check용 (섹션 0).
SANITY_CHECK_FILTERS = ("non_hate", "all")


def pred_path(outputs_root, seed, condition, modality):
    return os.path.join(outputs_root, f"seed{seed}", f"test_predictions_{condition}_{modality}.csv")


def _ensure_group_columns(df):
    """구버전 predictions_csv(image_group_id/text_group_id가 없는)를 만나면 id로
    폴백한다 -- 매칭 구조는 없다고 가정하는 것과 같아 안전한 기본값이다."""
    if "image_group_id" not in df.columns:
        df["image_group_id"] = df["id"]
    if "text_group_id" not in df.columns:
        df["text_group_id"] = df["id"]
    return df


def load_long_frame(outputs_root, seeds, conditions=CONDITIONS, modalities=MODALITIES):
    """모든 (seed, condition, modality) 조합의 예측 csv를 long-format 하나로 합친다.
    누락된 파일은 경고만 남기고 건너뛴다(아직 안 돌린 seed/조건이 있어도 있는 것만으로
    분석을 진행할 수 있게)."""
    rows = []
    for seed in seeds:
        for condition, (h_t, h_i) in conditions.items():
            for modality in modalities:
                path = pred_path(outputs_root, seed, condition, modality)
                if not os.path.exists(path):
                    warnings.warn(f"[누락] {path} 없음 -- 이 (seed={seed}, condition={condition}, "
                                  f"modality={modality}) 조합은 분석에서 빠집니다.")
                    continue
                df = pd.read_csv(path)
                _ensure_group_columns(df)
                df["seed"] = seed
                df["condition"] = condition
                df["modality"] = modality
                df["H_T"] = h_t
                df["H_I"] = h_i
                # pred_hate = 모델이 "혐오"로 예측했는지(원래 지표, 조건별 탐지율).
                # correct = 이 모달리티 입력 기준 정답(expected_hate)과 일치하는지.
                # full에서는 세 조건 모두 정답이 혐오라 두 값이 같다.
                df["pred_hate"] = (df["pred_label"] == "hate").astype(int)
                df["expected_hate"] = expected_hate(modality, h_t, h_i)
                df["correct"] = (df["pred_hate"] == df["expected_hate"]).astype(int)
                rows.append(df[["seed", "condition", "modality", "H_T", "H_I",
                                 "id", "image_group_id", "text_group_id",
                                 "score", "prob_hate", "pred_hate", "expected_hate", "correct"]])
    if not rows:
        raise FileNotFoundError(
            f"{outputs_root} 아래에서 예측 csv를 하나도 찾지 못했습니다. run_multi_seed.sh로 "
            "먼저 학습·평가를 돌렸는지, --outputs-root/--seeds가 맞는지 확인하세요."
        )
    return pd.concat(rows, ignore_index=True)


def load_sanity_frame(outputs_root, seeds, filters=SANITY_CHECK_FILTERS, modality="full"):
    """non_hate/all 채점 결과(전체 accuracy/FPR 확인용). 매칭 설계엔 참여하지 않는다."""
    rows = []
    for seed in seeds:
        for f in filters:
            path = pred_path(outputs_root, seed, f, modality)
            if not os.path.exists(path):
                warnings.warn(f"[누락] {path} 없음 -- sanity check에서 {f} 조건이 빠집니다.")
                continue
            df = pd.read_csv(path)
            df["seed"] = seed
            df["eval_filter"] = f
            y_true = (df["true_target"] == "혐오").astype(int)
            y_pred = (df["pred_label"] == "hate").astype(int)
            rows.append(pd.DataFrame({
                "seed": [seed], "eval_filter": [f], "n": [len(df)],
                "accuracy": [(y_true == y_pred).mean()],
                # non_hate 필터에서는 y_true가 전부 0이므로 fpr = 1 - accuracy(=오탐률).
                "fpr_or_na": [float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).any() else float("nan")],
                "always_hate_rate": [float((y_pred == 1).mean())],
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# 1) Mixed-effects logistic regression: crossed random effects (image, text)
#    주효과만 (00 없어 상호작용 항 제외)
# --------------------------------------------------------------------------
def fit_mixed_effects(long_df, modality="full"):
    """image_group_id와 text_group_id를 **교차(crossed) random intercept 두 개**로 둔
    mixed-effects logistic regression (correct ~ H_T + H_I). statsmodels의
    BinomialBayesMixedGLM은 vc_formula 딕셔너리에 키를 여러 개 주면 그만큼의 crossed
    variance component를 만들어준다 -- 이게 R lme4의 `(1|image_group) + (1|text_group)`
    에 대응한다. 단일 group_id로 합치면(예: 완전 연결요소) data_prep.py --case-pairs의
    전면적 재사용 때문에 표본 대부분이 하나의 거대 클러스터가 되어버려(모듈 docstring
    참고) random effect 추정이 무의미해지므로, 반드시 이렇게 두 축을 분리해야 한다.

    00 조건이 없어 H_T:H_I 상호작용 항은 여기 포함하지 않는다(모듈 docstring 참고).
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    sub = long_df[long_df["modality"] == modality].copy()
    if sub["image_group_id"].nunique() < 2 or sub["text_group_id"].nunique() < 2:
        raise ValueError(f"modality={modality}: crossed random effect를 추정하기엔 "
                          "image_group_id/text_group_id 종류가 너무 적습니다.")

    model = BinomialBayesMixedGLM.from_formula(
        "correct ~ H_T + H_I",
        vc_formulas={
            "image": "0 + C(image_group_id)",
            "text": "0 + C(text_group_id)",
        },
        data=sub,
    )
    result = model.fit_vb()
    fe_names = list(result.model.exog_names)
    fe_means = result.fe_mean
    fe_sd = result.fe_sd
    summary_rows = [
        {"term": name, "coef": float(mean), "sd": float(sd)}
        for name, mean, sd in zip(fe_names, fe_means, fe_sd)
    ]
    return pd.DataFrame(summary_rows)


# --------------------------------------------------------------------------
# 2) Pairwise two-way cluster bootstrap (11 vs 10, 11 vs 01, 10 vs 01)
# --------------------------------------------------------------------------
def _two_way_bootstrap_weights(rng, sub):
    """image_group_id와 text_group_id를 **독립적으로** 복원추출(with replacement)해서,
    각 원본 표본에 (해당 이미지가 뽑힌 횟수) x (해당 텍스트가 뽑힌 횟수)를 가중치로
    부여한다. 이는 two-way(다차원) cluster bootstrap의 표준적인 곱셈 가중치 방식으로,
    같은 이미지를 공유하는 표본들과 같은 텍스트를 공유하는 표본들 양쪽의 비독립성을
    각각 반영하면서도, 완전 연결요소로 묶었을 때 생기는 거대 클러스터 붕괴 문제를
    피한다.

    이전 버전의 버그: `sub[sub["group_id"].isin(sampled_groups)]`는 복원추출로 두 번
    뽑힌 그룹도 딱 한 번만 포함시켜서(집합 멤버십 검사라 중복이 사라짐) 사실상
    복원추출이 아니라 부분표본추출이 되어 있었다. 이 함수는 등장 "횟수"를 가중치로
    직접 쓰기 때문에 그 문제가 없다."""
    img_ids = sub["image_group_id"].unique()
    txt_ids = sub["text_group_id"].unique()

    img_draw = rng.choice(img_ids, size=len(img_ids), replace=True)
    txt_draw = rng.choice(txt_ids, size=len(txt_ids), replace=True)

    img_mult = pd.Series(img_draw).value_counts()
    txt_mult = pd.Series(txt_draw).value_counts()

    w_img = sub["image_group_id"].map(img_mult).fillna(0).to_numpy()
    w_txt = sub["text_group_id"].map(txt_mult).fillna(0).to_numpy()
    return w_img * w_txt


def cluster_bootstrap_diff(long_df, cond_a, cond_b, modality="full", metric="correct",
                            n_boot=2000, seed=0):
    """두 조건 간 metric(기본 correct=accuracy) 평균 차이의 two-way cluster bootstrap
    95% CI. image_group_id/text_group_id를 각각 독립적으로 복원추출해서(위
    _two_way_bootstrap_weights 참고) 두 매칭 축의 비독립성을 모두 반영한다."""
    rng = np.random.RandomState(seed)
    sub = long_df[(long_df["modality"] == modality) & (long_df["condition"].isin([cond_a, cond_b]))].copy()

    if sub.empty:
        return {"cond_a": cond_a, "cond_b": cond_b, "modality": modality, "metric": metric,
                "diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_images": 0, "n_texts": 0}

    def weighted_group_diff(frame, weights):
        a_mask = (frame["condition"] == cond_a).to_numpy()
        b_mask = (frame["condition"] == cond_b).to_numpy()
        wa, wb = weights[a_mask], weights[b_mask]
        if wa.sum() == 0 or wb.sum() == 0:
            return np.nan
        vals = frame[metric].to_numpy()
        mean_a = np.sum(vals[a_mask] * wa) / wa.sum()
        mean_b = np.sum(vals[b_mask] * wb) / wb.sum()
        return mean_a - mean_b

    point = weighted_group_diff(sub, np.ones(len(sub)))

    boot_diffs = []
    for _ in range(n_boot):
        weights = _two_way_bootstrap_weights(rng, sub)
        d = weighted_group_diff(sub, weights)
        if not np.isnan(d):
            boot_diffs.append(d)

    if not boot_diffs:
        return {"cond_a": cond_a, "cond_b": cond_b, "modality": modality, "metric": metric,
                "diff": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_images": sub["image_group_id"].nunique(), "n_texts": sub["text_group_id"].nunique()}

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return {
        "cond_a": cond_a, "cond_b": cond_b, "modality": modality, "metric": metric,
        "diff": point, "ci_low": ci_low, "ci_high": ci_high,
        "significant": not (ci_low <= 0 <= ci_high),
        "n_images": sub["image_group_id"].nunique(), "n_texts": sub["text_group_id"].nunique(),
    }


def all_pairwise_contrasts(long_df, modality="full", metric="correct", n_boot=2000, seed=0):
    rows = []
    for cond_a, cond_b in itertools.combinations(CONDITIONS.keys(), 2):
        rows.append(cluster_bootstrap_diff(long_df, cond_a, cond_b, modality, metric, n_boot, seed))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3) Multi-seed 집계
# --------------------------------------------------------------------------
def multi_seed_summary(long_df, modality="full", metric="correct"):
    """조건별 >=5 seed 평균/표준편차. seed가 5개 미만이면 경고만 남긴다(그래도 계산은 함)."""
    n_seeds = long_df["seed"].nunique()
    if n_seeds < 5:
        warnings.warn(f"seed가 {n_seeds}개뿐입니다 -- 방법론 권장(5개 이상)에 못 미칩니다. "
                       "run_multi_seed.sh로 나머지 seed를 더 돌리세요.")
    sub = long_df[long_df["modality"] == modality]
    per_seed = sub.groupby(["condition", "seed"])[metric].mean().reset_index()
    summary = per_seed.groupby("condition")[metric].agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n_seeds"})
    return summary


# --------------------------------------------------------------------------
# 4) McNemar (같은 표본에 대한 paired 예측 비교) + 다중비교 보정
# --------------------------------------------------------------------------
def mcnemar_modality_pairs(long_df, modality_a="full", modality_b="text_only"):
    """조건별로, 같은 (seed, id) 표본에 대한 modality_a vs modality_b 예측을 짝지어
    McNemar 검정을 수행한다 -- "이미지를 없애면(text_only) 정오답이 바뀌는가"처럼 같은
    표본에 대한 paired 비교에 쓴다. 여러 조건 x seed 조합에 걸쳐 검정을 반복하므로
    Benjamini-Hochberg(FDR)로 다중비교를 보정한다.

    correct는 모달리티마다 자기 입력 기준 정답(expected_hate)으로 계산돼 있으므로, 여기서
    비교하는 것은 "각 모달리티가 자기 입력에 대해 맞게 답했는가"의 paired 차이다."""
    results = []
    for condition in long_df["condition"].unique():
        for seed in long_df["seed"].unique():
            a = long_df[(long_df["condition"] == condition) & (long_df["seed"] == seed)
                        & (long_df["modality"] == modality_a)][["id", "correct"]]
            b = long_df[(long_df["condition"] == condition) & (long_df["seed"] == seed)
                        & (long_df["modality"] == modality_b)][["id", "correct"]]
            if a.empty or b.empty:
                continue
            merged = a.merge(b, on="id", suffixes=("_a", "_b"))
            if merged.empty:
                continue
            n_both_correct = int(((merged["correct_a"] == 1) & (merged["correct_b"] == 1)).sum())
            n_a_only = int(((merged["correct_a"] == 1) & (merged["correct_b"] == 0)).sum())
            n_b_only = int(((merged["correct_a"] == 0) & (merged["correct_b"] == 1)).sum())
            n_neither = int(((merged["correct_a"] == 0) & (merged["correct_b"] == 0)).sum())
            table = [[n_both_correct, n_a_only], [n_b_only, n_neither]]
            test = mcnemar(table, exact=(n_a_only + n_b_only < 25))
            results.append({
                "condition": condition, "seed": seed,
                "modality_a": modality_a, "modality_b": modality_b,
                "n": len(merged), "n_a_only_correct": n_a_only, "n_b_only_correct": n_b_only,
                "statistic": test.statistic, "p_value": test.pvalue,
            })
    df = pd.DataFrame(results)
    if len(df):
        rejected, p_adj, _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["p_value_bh"] = p_adj
        df["significant_bh_0.05"] = rejected
    return df


# --------------------------------------------------------------------------
# 리포트 조립
# --------------------------------------------------------------------------
def build_report(outputs_root, seeds, n_boot, out_path):
    long_df = load_long_frame(outputs_root, seeds)
    sanity_df = load_sanity_frame(outputs_root, seeds)

    lines = ["# Matched-design 분석 결과\n"]
    lines.append(f"- seeds: {sorted(long_df['seed'].unique().tolist())}\n")
    lines.append("- 조건: hate(박나현, H_T=1,H_I=1) / image_hate_text_safe(조현서, H_T=0,H_I=1) / "
                  "text_hate_image_safe(정미정, H_T=1,H_I=0). **00(H_T=0,H_I=0) 없음 -- "
                  "H_T×H_I 상호작용 전체(Delta)는 추정 불가, 아래는 주효과 + pairwise 대비만.**\n")

    lines.append("\n## 0) Sanity check: non_hate/all 전체 accuracy·FPR\n")
    lines.append("세 혐오 조건만 보면 recall(탐지율)만 확인하게 되어 '무조건 혐오'라고 "
                  "답하는 퇴화 모델도 만점처럼 보일 수 있다. non_hate 채점의 accuracy가 "
                  "낮고 always_hate_rate가 1에 가까우면 그런 퇴화가 의심된다.\n")
    lines.append(sanity_df.to_markdown(index=False) if len(sanity_df) else "(non_hate/all 채점 결과 없음 -- "
                 "run_multi_seed.sh가 --eval-filter non_hate/all도 채점했는지 확인하세요.)")

    lines.append("\n\n## 1) Multi-seed 요약 (모달리티별)\n")
    lines.append("correct = 그 모달리티 입력 기준 정답과 일치 비율(full: OR 규칙, text_only: H_T, "
                  "image_only: H_I). pred_hate = '혐오'로 예측한 비율.\n")
    for modality in MODALITIES:
        if not (long_df["modality"] == modality).any():
            continue
        summary = multi_seed_summary(long_df, modality=modality, metric="correct")
        pred_rate = multi_seed_summary(long_df, modality=modality, metric="pred_hate")
        summary = summary.merge(pred_rate[["condition", "pred_hate_mean", "pred_hate_std"]], on="condition")
        summary["owner"] = summary["condition"].map(CONDITION_OWNER)
        summary["expected"] = summary["condition"].map(
            lambda c: "혐오" if expected_hate(modality, *CONDITIONS[c]) else "비혐오")
        lines.append(f"\n### {modality}\n")
        lines.append(summary.to_markdown(index=False))

    lines.append("\n\n## 2) Mixed-effects logistic regression "
                  "(correct ~ H_T + H_I, crossed random intercepts: image_group_id, text_group_id)\n")
    try:
        fe = fit_mixed_effects(long_df, modality="full")
        lines.append(fe.to_markdown(index=False))
    except Exception as e:  # pragma: no cover - statsmodels/optional dep 환경 문제 대비
        lines.append(f"(mixed-effects 적합 실패: {e})")

    lines.append("\n\n## 3) Pairwise two-way cluster bootstrap 대비 (95% CI, full modality)\n")
    contrasts = all_pairwise_contrasts(long_df, modality="full", metric="correct", n_boot=n_boot)
    lines.append(contrasts.to_markdown(index=False))

    lines.append("\n\n## 4) McNemar: full vs text-only (각 모달리티 입력 기준 정답 대비 정오답 변화)\n")
    mc_text = mcnemar_modality_pairs(long_df, "full", "text_only")
    lines.append(mc_text.to_markdown(index=False) if len(mc_text) else "(비교할 데이터 없음)")

    lines.append("\n\n## 5) McNemar: full vs image-only (각 모달리티 입력 기준 정답 대비 정오답 변화)\n")
    mc_image = mcnemar_modality_pairs(long_df, "full", "image_only")
    lines.append(mc_image.to_markdown(index=False) if len(mc_image) else "(비교할 데이터 없음)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(l) for l in lines) + "\n")
    print(f"리포트 저장: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs",
                     help="run_multi_seed.sh가 seed<N>/ 하위에 저장한 예측 csv들의 상위 폴더.")
    ap.add_argument("--seeds", default="1,2,3,4,5",
                     help="쉼표로 구분한 seed 목록 (예: 1,2,3,4,5). 5개 이상 권장.")
    ap.add_argument("--n-boot", type=int, default=2000, help="cluster bootstrap 재표본 횟수.")
    ap.add_argument("--report-out", default="outputs/matched_design_report.md")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    build_report(args.outputs_root, seeds, args.n_boot, args.report_out)


if __name__ == "__main__":
    main()
