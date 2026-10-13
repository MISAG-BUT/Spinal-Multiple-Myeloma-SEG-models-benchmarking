import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, shapiro, wilcoxon


METRICS = ["Dice", "F1"]
SINGLE_INPUT_IDS = list(range(709, 717))
LOO_IDS = list(range(700, 708))
GROUND_TRUTH_ID = 708
EXPECTED_EXAM_COUNT = 72
EXPECTED_PATIENT_COUNT = 67
EXPECTED_DATASET_IDS = set(range(700, 719))
REPEATED_EXAMS = {
    "Myel_012_a": "Myel_012",
    "Myel_012_b": "Myel_012",
    "Myel_018_a": "Myel_018",
    "Myel_018_b": "Myel_018",
    "Myel_023_a": "Myel_023",
    "Myel_023_b": "Myel_023",
    "Myel_043_a": "Myel_043",
    "Myel_043_b": "Myel_043",
    "Myel_052_a": "Myel_052",
    "Myel_052_b": "Myel_052",
}


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
        duplicated = selected.duplicated(["case_id", condition_column], keep=False)
        if duplicated.any():
            examples = selected.loc[duplicated, ["case_id", condition_column]].head()
            raise ValueError(
                f"Duplicate case/condition rows in {analysis_name}: "
                f"{examples.to_dict('records')}"
            )
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
        duplicated = selected.duplicated(["case_id", "dataset_id"], keep=False)
        if duplicated.any():
            examples = selected.loc[duplicated, ["case_id", "dataset_id"]].head()
            raise ValueError(
                f"Duplicate case/dataset rows in LOO analysis: "
                f"{examples.to_dict('records')}"
            )
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


def patient_id_from_case_id(case_id):
    """Return patient ID while retaining distinct IDs for non-repeat examinations."""
    name = os.path.basename(str(case_id).strip())
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        stem = name[:-7]
    elif lower_name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = name
    return REPEATED_EXAMS.get(stem, stem)


def validate_patient_level_source(longi):
    required_columns = {"case_id", "dataset_id", "Dice", "F1"}
    missing_columns = required_columns.difference(longi.columns)
    if missing_columns:
        raise ValueError(f"Missing patient-level source columns: {sorted(missing_columns)}")

    data = longi.copy()
    if data["case_id"].isna().any():
        raise ValueError("case_id contains missing values.")
    data["case_id"] = data["case_id"].astype(str)
    data["dataset_id"] = pd.to_numeric(data["dataset_id"], errors="coerce")
    if data["dataset_id"].isna().any() or not np.equal(
        data["dataset_id"], np.floor(data["dataset_id"])
    ).all():
        raise ValueError("dataset_id must contain integer model IDs.")
    data["dataset_id"] = data["dataset_id"].astype(int)
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        if not np.isfinite(data[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"{metric} contains missing or non-finite values.")

    dataset_ids = set(data["dataset_id"].unique())
    if dataset_ids != EXPECTED_DATASET_IDS:
        raise ValueError(
            f"Expected dataset IDs 700-718, got {sorted(dataset_ids)}."
        )
    if data.duplicated(["case_id", "dataset_id"]).any():
        examples = data.loc[
            data.duplicated(["case_id", "dataset_id"], keep=False),
            ["case_id", "dataset_id"],
        ].head()
        raise ValueError(
            f"Duplicate case_id/dataset_id rows found: {examples.to_dict('records')}"
        )

    cases = sorted(data["case_id"].unique())
    if len(cases) != EXPECTED_EXAM_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EXAM_COUNT} unique examinations, got {len(cases)}."
        )
    case_sets = {
        dataset_id: set(group["case_id"])
        for dataset_id, group in data.groupby("dataset_id")
    }
    baseline_cases = case_sets[min(EXPECTED_DATASET_IDS)]
    if any(case_set != baseline_cases for case_set in case_sets.values()):
        raise ValueError("The 19 models do not contain the same 72 examination IDs.")

    data["patient_id"] = data["case_id"].map(patient_id_from_case_id)
    visit_counts = data[["case_id", "patient_id"]].drop_duplicates().groupby(
        "patient_id"
    )["case_id"].nunique()
    repeated_patient_ids = {
        f"Myel_{patient_number}"
        for patient_number in ("012", "018", "023", "043", "052")
    }
    actual_repeated = set(visit_counts[visit_counts == 2].index)
    if actual_repeated != repeated_patient_ids:
        raise ValueError(
            "Expected exactly the five known patients to have two examinations; "
            f"got {sorted(actual_repeated)}."
        )
    if (visit_counts > 2).any() or (visit_counts == 1).sum() != 62:
        raise ValueError("Unexpected examination counts after patient ID extraction.")
    if len(visit_counts) != EXPECTED_PATIENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PATIENT_COUNT} unique patients, got {len(visit_counts)}."
        )

    return data


def aggregate_patient_metrics(longi):
    data = validate_patient_level_source(longi)
    patient_data = data.groupby(
        ["patient_id", "dataset_id"],
        as_index=False,
        sort=True,
    )[METRICS].mean()
    patient_data["case_id"] = patient_data["patient_id"]

    counts = patient_data.groupby("dataset_id")["patient_id"].nunique()
    if set(counts.index) != EXPECTED_DATASET_IDS or not (counts == EXPECTED_PATIENT_COUNT).all():
        raise ValueError(
            "Patient aggregation did not produce exactly 67 values for each model."
        )
    if patient_data.duplicated(["patient_id", "dataset_id"]).any():
        raise ValueError("Patient aggregation produced duplicate patient/model rows.")
    return patient_data


def validate_complete_patient_pairs(data, dataset_ids):
    selected = data[data["dataset_id"].isin(dataset_ids)]
    if selected.duplicated(["patient_id", "dataset_id"]).any():
        raise ValueError("Duplicate patient/model rows before paired statistical tests.")
    expected_patients = set(data["patient_id"].unique())
    for dataset_id in dataset_ids:
        model_patients = set(
            selected.loc[selected["dataset_id"] == dataset_id, "patient_id"]
        )
        if model_patients != expected_patients:
            raise ValueError(
                f"Dataset {dataset_id} does not have complete pairing for all patients."
            )


def _comparison_table(original, patient, keys):
    columns = keys + ["p_value", "holm_p_value", "rank_biserial_effect"]
    old = original[columns].rename(columns={
        "p_value": "p_value_72_examinations",
        "holm_p_value": "holm_p_value_72_examinations",
        "rank_biserial_effect": "rank_biserial_effect_72_examinations",
    })
    new = patient[columns].rename(columns={
        "p_value": "p_value_67_patients",
        "holm_p_value": "holm_p_value_67_patients",
        "rank_biserial_effect": "rank_biserial_effect_67_patients",
    })
    comparison = old.merge(new, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not comparison["_merge"].eq("both").all():
        raise ValueError("Could not pair all original and patient-level statistical results.")
    comparison = comparison.drop(columns="_merge")
    for column in (
        "holm_p_value_72_examinations",
        "holm_p_value_67_patients",
    ):
        if comparison[column].isna().any():
            raise ValueError(f"Missing adjusted p-values in {column}.")
    comparison["significant_72_examinations"] = (
        comparison["holm_p_value_72_examinations"] < 0.05
    )
    comparison["significant_67_patients"] = (
        comparison["holm_p_value_67_patients"] < 0.05
    )
    comparison["conclusion_changed"] = (
        comparison["significant_72_examinations"]
        != comparison["significant_67_patients"]
    )
    return comparison


def build_patient_level_sensitivity_tables(longi, original_tables):
    patient_data = aggregate_patient_metrics(longi)
    validate_complete_patient_pairs(patient_data, SINGLE_INPUT_IDS)
    validate_complete_patient_pairs(
        patient_data,
        [GROUND_TRUTH_ID, *LOO_IDS],
    )

    single_input = patient_data[
        patient_data["dataset_id"].isin(SINGLE_INPUT_IDS)
    ].copy()
    single_input["condition"] = single_input["dataset_id"].map(
        lambda dataset_id: f"Dataset_{dataset_id}"
    )
    conditions = [f"Dataset_{dataset_id}" for dataset_id in SINGLE_INPUT_IDS]
    omnibus, pairwise, normality = run_paired_analysis(
        single_input,
        conditions,
        "condition",
        "single_input_models_709_716_patient_level",
    )
    loo_comparisons, loo_differences, loo_zero_tests, loo_normality = (
        run_loo_comparisons(patient_data)
    )

    for table in (omnibus, pairwise):
        if "n_complete_patients" in table.columns:
            table.rename(columns={"n_complete_patients": "n_patients"}, inplace=True)
    normality.rename(columns={"n": "n_patients"}, inplace=True)

    comparison_single = _comparison_table(
        original_tables["single_input_pairwise_tests.csv"],
        pairwise,
        ["metric", "condition_1", "condition_2"],
    )
    comparison_loo = _comparison_table(
        original_tables["loo_differences_vs_zero_tests.csv"],
        loo_zero_tests,
        ["metric", "condition"],
    )

    tables = {
        "patient_level_aggregated_values.csv": patient_data,
        "single_input_omnibus_tests.csv": omnibus,
        "single_input_pairwise_tests.csv": pairwise,
        "single_input_normality.csv": normality,
        "loo_vs_all_together_pairwise_tests.csv": loo_comparisons.rename(
            columns={"n_complete_patients": "n_patients"}
        ),
        "loo_patient_level_differences.csv": loo_differences,
        "loo_differences_vs_zero_tests.csv": loo_zero_tests,
        "loo_normality_tests.csv": loo_normality,
        "single_input_72_vs_67.csv": comparison_single,
        "loo_72_vs_67.csv": comparison_loo,
    }
    summary = {
        "exam_count": len(longi["case_id"].unique()),
        "patient_count": patient_data["patient_id"].nunique(),
        "single_input": {},
        "loo_significant": {},
        "changed_conclusions": int(
            comparison_single["conclusion_changed"].sum()
            + comparison_loo["conclusion_changed"].sum()
        ),
    }
    for metric in METRICS:
        old_metric = original_tables["single_input_pairwise_tests.csv"]
        summary["single_input"][metric] = {
            "72_examinations": int(
                (
                    (old_metric["metric"] == metric)
                    & (old_metric["holm_p_value"] < 0.05)
                ).sum()
            ),
            "67_patients": int(
                (
                    (pairwise["metric"] == metric)
                    & (pairwise["holm_p_value"] < 0.05)
                ).sum()
            ),
        }
        summary["loo_significant"][metric] = {}
        for label, result in (
            ("72_examinations", original_tables["loo_differences_vs_zero_tests.csv"]),
            ("67_patients", loo_zero_tests),
        ):
            significant = result[
                (result["metric"] == metric) & (result["holm_p_value"] < 0.05)
            ]["condition"].tolist()
            summary["loo_significant"][metric][label] = significant

    return tables, summary


def print_patient_sensitivity_summary(summary):
    print(
        "Patient-level sensitivity summary: "
        f"{summary['exam_count']} examinations, "
        f"{summary['patient_count']} patients."
    )
    for metric in METRICS:
        counts = summary["single_input"][metric]
        print(
            f"  Single-input {metric} significant comparisons: "
            f"72 examinations={counts['72_examinations']}, "
            f"67 patients={counts['67_patients']}."
        )
        for analysis in ("72_examinations", "67_patients"):
            significant = summary["loo_significant"][metric][analysis]
            print(f"  LOO {metric} significant ({analysis}): {significant or 'none'}.")
    print(f"  Changed statistical conclusions: {summary['changed_conclusions']}.")


if __name__ == "__main__":
    from generate_suplementary_tables import main

    main()
