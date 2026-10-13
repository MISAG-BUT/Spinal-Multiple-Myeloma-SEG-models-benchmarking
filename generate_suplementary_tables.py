import os
import re

import pandas as pd

import analyze_gpu_usage_final
import generate_longi_dice_f1_boxplots as figures
import statistical_analysis


ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
OUTPUT_DIR = os.path.join(ROOT, "results_final", "tables_final")
ANALYSES = list(figures.THRESHOLD_ANALYSES.values())
RESULTS_DIR = os.path.join(ROOT, "results_final")
IOU_F1_SUMMARY_FILE = os.path.join(
    ROOT,
    "iou_threshold_analysis_full_models",
    figures.ANALYSIS,
    "iou_threshold_f1_summary.csv",
)


def extract_dataset_id(name):
    match = re.search(r"Dataset(\d+)", str(name))
    return int(match.group(1)) if match else 9999


def save_table(df, filename, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path


def save_workbook(sheets, filename, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved: {output_path}")
    return output_path


def generate_model_summary_tables():
    all_rows = []
    for analysis in ANALYSES:
        file_path = os.path.join(RESULTS_DIR, analysis, "results_all_folds.csv")
        if not os.path.exists(file_path):
            continue
        analysis_df = pd.read_csv(file_path)
        analysis_df["analysis"] = analysis
        all_rows.append(analysis_df)
    if not all_rows:
        raise FileNotFoundError("No results files found for supplementary tables")

    df = pd.concat(all_rows, ignore_index=True)
    df["dataset_id"] = df["model_name"].apply(extract_dataset_id)
    df = df.sort_values(["dataset_id", "model_type", "fold"])
    summary = df.groupby(
        ["analysis", "model_name", "model_type"]
    ).agg(
        Dice_mean=("Dice", "mean"),
        Dice_std=("Dice", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        NSD_mean=("NSD", "mean"),
        NSD_std=("NSD", "std"),
    ).reset_index()
    summary["dataset_id"] = summary["model_name"].apply(extract_dataset_id)
    summary = summary.sort_values(["analysis", "dataset_id"])

    save_table(df, "MM_full_results_all_thresholds.csv")
    save_workbook({"results": df}, "MM_full_results_all_thresholds.xlsx")
    save_table(summary, "MM_summary_table.csv")
    save_workbook({"summary": summary}, "MM_summary_table.xlsx")


def generate_patient_level_tables():
    patient_df = figures.load_patient_rows()
    threshold_df = figures.load_threshold_patient_rows()
    per_patient_dir = os.path.join(OUTPUT_DIR, "per_patient")
    save_table(
        patient_df,
        "longi_summary_all_per_patient_values.csv",
        per_patient_dir,
    )
    save_table(
        threshold_df,
        "threshold_comparison_per_patient_values.csv",
        per_patient_dir,
    )
    return patient_df


def generate_feature_importance_tables():
    feature_dir = os.path.join(OUTPUT_DIR, "feature_importance")
    for threshold_label, analysis_name in figures.THRESHOLD_ANALYSES.items():
        df, df_mean, importance_df = figures.compute_feature_importance(analysis_name)
        if df is None:
            print(f"No feature importance results for threshold: {threshold_label}")
            continue

        threshold_dir = os.path.join(feature_dir, threshold_label)
        save_table(df, "results_all_folds.csv", threshold_dir)
        save_table(df_mean, "results_model_means.csv", threshold_dir)
        save_table(importance_df, "feature_importance.csv", threshold_dir)
        sheets = {
            "all_folds": df,
            "model_means": df_mean,
            "feature_importance": importance_df,
        }
        if not importance_df.empty:
            label_map = {
                "Dataset708": "All together",
                "Dataset717": "All VMI",
                "Dataset718": "All CaSupp",
            }
            dice_heatmap = importance_df.pivot_table(
                index="zero_input_channel",
                columns="dataset",
                values="Dice_drop",
            ).rename(columns=label_map)
            f1_heatmap = importance_df.pivot_table(
                index="zero_input_channel",
                columns="dataset",
                values="F1_drop",
            ).rename(columns=label_map)
            save_table(
                dice_heatmap.reset_index(),
                "feature_importance_heatmap_Dice.csv",
                threshold_dir,
            )
            save_table(
                f1_heatmap.reset_index(),
                "feature_importance_heatmap_F1.csv",
                threshold_dir,
            )
            sheets["heatmap_Dice"] = dice_heatmap.reset_index()
            sheets["heatmap_F1"] = f1_heatmap.reset_index()
        save_workbook(sheets, "feature_importance_summary.xlsx", threshold_dir)


def generate_statistical_tables(patient_df):
    stats_dir = os.path.join(OUTPUT_DIR, "statistical_analysis")
    original_tables = statistical_analysis.build_statistical_tables(patient_df)
    for filename, table in original_tables.items():
        save_table(table, filename, stats_dir)
    sensitivity_tables, summary = statistical_analysis.build_patient_level_sensitivity_tables(
        patient_df,
        original_tables,
    )
    sensitivity_dir = os.path.join(stats_dir, "patient_level_sensitivity")
    for filename, table in sensitivity_tables.items():
        save_table(table, filename, sensitivity_dir)
    statistical_analysis.print_patient_sensitivity_summary(summary)


def generate_iou_input_table():
    if not os.path.exists(IOU_F1_SUMMARY_FILE):
        raise FileNotFoundError(f"IoU threshold summary not found: {IOU_F1_SUMMARY_FILE}")
    df = pd.read_csv(IOU_F1_SUMMARY_FILE)
    required_columns = {"model", "iou_threshold", "F1_mean", "F1_std"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in IoU threshold summary: {sorted(missing_columns)}")
    save_table(
        df,
        "iou_threshold_f1_summary.csv",
        os.path.join(OUTPUT_DIR, "iou_threshold_analysis"),
    )


def generate_gpu_tables():
    gpu_dir = os.path.join(OUTPUT_DIR, "gpu_analysis")
    for filename, table in analyze_gpu_usage_final.build_gpu_tables().items():
        save_table(table, filename, gpu_dir)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_model_summary_tables()
    patient_df = generate_patient_level_tables()
    generate_feature_importance_tables()
    generate_statistical_tables(patient_df)
    generate_iou_input_table()
    generate_gpu_tables()
    print(f"Generated supplementary tables in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
