import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, shapiro, wilcoxon


METRICS = ["Dice", "F1"]
SINGLE_INPUT_IDS = list(range(709, 717))
LOO_IDS = list(range(700, 708))
GROUND_TRUTH_ID = 708


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(p_values[valid])]

    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted_value = (len(order) - rank) * p_values[index]
        running_max = max(running_max, adjusted_value)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def significance_label(p_value):
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def paired_rank_biserial(x, y):
    """Effect size for paired data: positive values favor x over y."""
    differences = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if len(differences) == 0:
        return 0.0

    ranks = rankdata(np.abs(differences), method="average")
    rank_sum = ranks.sum()
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / rank_sum)


def paired_wilcoxon(x, y):
    differences = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0 or np.allclose(differences, 0):
        return np.nan
    return float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)


def shapiro_p_value(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or len(values) > 5000:
        return np.nan
    return float(shapiro(values).pvalue)


def run_paired_analysis(data, conditions, condition_column, analysis_name):
    omnibus_rows = []
    pairwise_rows = []
    normality_rows = []
    raw_pairwise_rows = []

    for metric in METRICS:
        selected = data[data[condition_column].isin(conditions)].copy()
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected = selected.dropna(subset=["case_id", condition_column, metric])

        # Keep only complete patient blocks for a valid paired comparison.
        complete_cases = (
            selected.groupby("case_id")[condition_column]
            .nunique()
            .loc[lambda values: values == len(conditions)]
            .index
        )
        complete = selected[selected["case_id"].isin(complete_cases)]
        wide = complete.pivot_table(
            index="case_id",
            columns=condition_column,
            values=metric,
            aggfunc="first",
        ).reindex(columns=conditions).dropna()

        n = len(wide)
        for condition in conditions:
            normality_p = shapiro_p_value(wide[condition].to_numpy())
            normality_rows.append({
                "analysis": analysis_name,
                "metric": metric,
                "condition": condition,
                "n": n,
                "shapiro_p_value": normality_p,
                "normal_distribution": (
                    "yes" if pd.notna(normality_p) and normality_p >= 0.05
                    else "no" if pd.notna(normality_p) else "not tested"
                ),
            })
        if n >= 2 and len(conditions) >= 3:
            statistic, omnibus_p = friedmanchisquare(
                *(wide[condition].to_numpy() for condition in conditions)
            )
        else:
            statistic, omnibus_p = np.nan, np.nan

        omnibus_rows.append({
            "analysis": analysis_name,
            "metric": metric,
            "n_complete_patients": n,
            "n_conditions": len(conditions),
            "test_used": "friedman" if n >= 2 and len(conditions) >= 3 else "not tested",
            "test_statistic": statistic,
            "p_value": omnibus_p,
            "significance": significance_label(omnibus_p),
        })

        for first_condition, second_condition in itertools.combinations(conditions, 2):
            if n >= 2:
                first_values = wide[first_condition].to_numpy()
                second_values = wide[second_condition].to_numpy()
                raw_p = paired_wilcoxon(first_values, second_values)
                effect = paired_rank_biserial(first_values, second_values)
            else:
                raw_p, effect = np.nan, np.nan

            raw_pairwise_rows.append({
                "analysis": analysis_name,
                "metric": metric,
                "condition_1": first_condition,
                "condition_2": second_condition,
                "n_complete_patients": n,
                "test_used": "wilcoxon" if n >= 2 else "not tested",
                "p_value": raw_p,
                "rank_biserial_effect": effect,
            })

    raw_pairwise = pd.DataFrame(raw_pairwise_rows)
    if not raw_pairwise.empty:
        raw_pairwise["holm_p_value"] = np.nan
        for metric, indices in raw_pairwise.groupby("metric").groups.items():
            raw_pairwise.loc[indices, "holm_p_value"] = holm_adjust(
                raw_pairwise.loc[indices, "p_value"].to_numpy()
            )
        raw_pairwise["significance"] = raw_pairwise["holm_p_value"].map(significance_label)
        pairwise_rows = raw_pairwise.to_dict("records")

    return (
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(normality_rows),
    )


def run_loo_comparisons(longi):
    comparison_rows = []
    difference_rows = []
    normality_rows = []

    for metric in METRICS:
        selected = longi[longi["dataset_id"].isin([GROUND_TRUTH_ID, *LOO_IDS])].copy()
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected = selected.dropna(subset=["case_id", "dataset_id", metric])
        wide = selected.pivot_table(
            index="case_id",
            columns="dataset_id",
            values=metric,
            aggfunc="first",
        ).reindex(columns=[GROUND_TRUTH_ID, *LOO_IDS])
        for loo_id in LOO_IDS:
            paired = wide[[GROUND_TRUTH_ID, loo_id]].dropna()
            ground_truth_values = paired[GROUND_TRUTH_ID].to_numpy()
            loo_values = paired[loo_id].to_numpy()
            paired_differences = loo_values - ground_truth_values
            normality_p = shapiro_p_value(paired_differences)
            normality_rows.append({
                "analysis": "loo_paired_differences",
                "metric": metric,
                "condition": f"Dataset_{loo_id}",
                "difference": f"Dataset_{loo_id} - Dataset_{GROUND_TRUTH_ID}",
                "n_patients": len(paired_differences),
                "normality_test": "shapiro_wilk",
                "alpha": 0.05,
                "shapiro_p_value": normality_p,
                "normality_assessment": (
                    "not_rejected_at_0.05" if pd.notna(normality_p) and normality_p >= 0.05
                    else "rejected_at_0.05" if pd.notna(normality_p)
                    else "not_tested"
                ),
            })
            comparison_rows.append({
                "analysis": "loo_vs_all_together",
                "metric": metric,
                "condition_1": f"Dataset_{GROUND_TRUTH_ID}",
                "condition_2": f"Dataset_{loo_id}",
                "n_complete_patients": len(paired),
                "test_used": "paired_wilcoxon" if len(paired) >= 2 else "not tested",
                "p_value": paired_wilcoxon(ground_truth_values, loo_values)
                if len(paired) >= 2 else np.nan,
                "rank_biserial_effect": paired_rank_biserial(
                    ground_truth_values, loo_values
                ) if len(paired) >= 2 else np.nan,
            })

            difference = paired[loo_id] - paired[GROUND_TRUTH_ID]
            for case_id, value in difference.items():
                difference_rows.append({
                    "case_id": case_id,
                    "dataset_id": loo_id,
                    "condition": f"Dataset_{loo_id}",
                    f"{metric}_diff": value,
                })

    comparison_tests = pd.DataFrame(comparison_rows)
    if not comparison_tests.empty:
        comparison_tests["holm_p_value"] = np.nan
        for _, indices in comparison_tests.groupby("metric").groups.items():
            comparison_tests.loc[indices, "holm_p_value"] = holm_adjust(
                comparison_tests.loc[indices, "p_value"].to_numpy()
            )
        comparison_tests["significance"] = comparison_tests["holm_p_value"].map(
            significance_label
        )

    differences = pd.DataFrame(difference_rows)
    if not differences.empty:
        differences = differences.groupby(
            ["case_id", "dataset_id", "condition"], as_index=False
        ).first()

    zero_rows = []
    for metric in METRICS:
        difference_column = f"{metric}_diff"
        for loo_id in LOO_IDS:
            values = pd.to_numeric(
                differences.loc[
                    differences["dataset_id"] == loo_id,
                    difference_column,
                ],
                errors="coerce",
            ).dropna().to_numpy()
            zero_rows.append({
                "analysis": "loo_difference_vs_zero",
                "metric": metric,
                "condition": f"Dataset_{loo_id}",
                "n_patients": len(values),
                "test_used": "one_sample_wilcoxon" if len(values) >= 2 else "not tested",
                "p_value": paired_wilcoxon(values, np.zeros(len(values)))
                if len(values) >= 2 else np.nan,
                "rank_biserial_effect": paired_rank_biserial(
                    values, np.zeros(len(values))
                ) if len(values) >= 2 else np.nan,
            })

    zero_tests = pd.DataFrame(zero_rows)
    if not zero_tests.empty:
        zero_tests["holm_p_value"] = np.nan
        for _, indices in zero_tests.groupby("metric").groups.items():
            zero_tests.loc[indices, "holm_p_value"] = holm_adjust(
                zero_tests.loc[indices, "p_value"].to_numpy()
            )
        zero_tests["significance"] = zero_tests["holm_p_value"].map(significance_label)

    return comparison_tests, differences, zero_tests, pd.DataFrame(normality_rows)


def build_statistical_tables(longi):
    longi["dataset_id"] = pd.to_numeric(longi["dataset_id"], errors="coerce")
    single_input = longi[longi["dataset_id"].isin(SINGLE_INPUT_IDS)].copy()
    single_conditions = [f"Dataset_{dataset_id}" for dataset_id in SINGLE_INPUT_IDS]
    single_input["condition"] = single_input["dataset_id"].map(
        lambda dataset_id: f"Dataset_{int(dataset_id)}"
    )
    single_omnibus, single_pairwise, single_normality = run_paired_analysis(
        single_input,
        single_conditions,
        "condition",
        "single_input_models_709_716",
    )
    loo_comparisons, loo_differences, loo_zero_tests, loo_normality = run_loo_comparisons(longi)

    return {
        "single_input_omnibus_tests.csv": single_omnibus,
        "single_input_pairwise_tests.csv": single_pairwise,
        "single_input_normality.csv": single_normality,
        "loo_vs_all_together_pairwise_tests.csv": loo_comparisons,
        "loo_patient_level_differences.csv": loo_differences,
        "loo_differences_vs_zero_tests.csv": loo_zero_tests,
        "loo_normality_tests.csv": loo_normality,
    }


if __name__ == "__main__":
    from generate_suplementary_tables import main

    main()
