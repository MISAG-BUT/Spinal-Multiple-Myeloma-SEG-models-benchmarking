import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch


ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
ANALYSIS = "longi_summary_all"
JSON_ROOT = os.path.join(ROOT, "full_models", ANALYSIS)
OUTPUT_DIR = os.path.join(ROOT, "results", "figures_final_new_3", "longi_summary_all_dice_f1")
STATISTICS_FILE = os.path.join(
    ROOT,
    "results",
    "figures_final_new_2",
    "statistical_analysis",
    "single_input_pairwise_tests.csv",
)
LOO_ZERO_STATISTICS_FILE = os.path.join(
    ROOT,
    "results",
    "figures_final_new_2",
    "statistical_analysis",
    "loo_differences_vs_zero_tests.csv",
)
THRESHOLD_ANALYSES = {
    "all": "longi_summary_all",
    "0.3cm": "longi_summary_larger_than_0_3_cubic_cm",
    "0.5cm": "longi_summary_larger_than_0_5_cubic_cm",
}
THRESHOLD_MODELS = {
    "ConvCT": "Dataset709_MM_Lesion_seg_just_ConvCT",
    "VMI40": "Dataset710_MM_Lesion_seg_just_VMI_40",
    "CaSupp25": "Dataset713_MM_Lesion_seg_just_CaSupp_25",
    "All together": "Dataset708_MM_Lesion_seg_all_together",
    "All VMI": "Dataset717_MM_Lesion_seg_all_VMI",
    "All CaSupp": "Dataset718_MM_Lesion_seg_all_CaSupp",
}
THRESHOLD_ORDER = ["all", "0.3cm", "0.5cm"]
BASE_MODELS = {
    "Dataset708": "Dataset708_MM_Lesion_seg_all_together",
    "Dataset717": "Dataset717_MM_Lesion_seg_all_VMI",
    "Dataset718": "Dataset718_MM_Lesion_seg_all_CaSupp",
}
IOU_F1_SUMMARY_FILE = os.path.join(
    ROOT,
    "iou_threshold_analysis_full_models",
    ANALYSIS,
    "iou_threshold_f1_summary.csv",
)

MODEL_IDS = list(range(700, 719))
GROUP_COLORS = {
    "LOO": "#72B0F1",
    "ConvCT": "#ff7f0e",
    "VMI": "#2ca02c",
    "CaSupp": "#d62728",
    "All together": "#9467bd",
}
DATASET_GROUP = {
    700: "LOO",
    701: "LOO",
    702: "LOO",
    703: "LOO",
    704: "LOO",
    705: "LOO",
    706: "LOO",
    707: "LOO",
    708: "All together",
    709: "ConvCT",
    710: "VMI",
    711: "VMI",
    712: "VMI",
    713: "CaSupp",
    714: "CaSupp",
    715: "CaSupp",
    716: "CaSupp",
    717: "VMI",
    718: "CaSupp",
}
DATASET_LABELS = {
    700: "LOO w/o ConvCT",
    701: "LOO w/o VMI 40",
    702: "LOO w/o VMI 80",
    703: "LOO w/o VMI 120",
    704: "LOO w/o CaSupp 25",
    705: "LOO w/o CaSupp 50",
    706: "LOO w/o CaSupp 75",
    707: "LOO w/o CaSupp 100",
    708: "All together",
    709: "ConvCT only",
    710: "VMI 40 keV only",
    711: "VMI 80 keV only",
    712: "VMI 120 keV only",
    713: "CaSupp 25 only",
    714: "CaSupp 50 only",
    715: "CaSupp 75 only",
    716: "CaSupp 100 only",
    717: "All VMI",
    718: "All CaSupp",
}


def extract_dataset_id(model_name):
    match = re.search(r"Dataset(\d+)", str(model_name))
    return int(match.group(1)) if match else None


def extract_case_id(item):
    for key in ("prediction_file", "reference_file"):
        file_path = item.get(key, "")
        if file_path:
            return os.path.splitext(os.path.basename(file_path))[0]
    return None


def load_patient_rows():
    rows = []
    if not os.path.isdir(JSON_ROOT):
        raise FileNotFoundError(f"Missing JSON root: {JSON_ROOT}")

    for model_dir in sorted(glob.glob(os.path.join(JSON_ROOT, "Dataset*"))):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        dataset_id = extract_dataset_id(model_name)
        if dataset_id not in MODEL_IDS:
            continue

        json_pattern = os.path.join(model_dir, f"*{ANALYSIS}*.json")
        for json_file in sorted(glob.glob(json_pattern)):
            if "ALL_FOLDS" in os.path.basename(json_file):
                continue
            with open(json_file) as file_handle:
                data = json.load(file_handle)

            for item in data.get("metric_per_case", []):
                case_metrics = item.get("metrics")
                if not isinstance(case_metrics, dict):
                    continue
                metric_entry = next(iter(case_metrics.values()), None)
                if not isinstance(metric_entry, dict):
                    continue
                rows.append({
                    "dataset_id": dataset_id,
                    "case_id": extract_case_id(item),
                    "Dice": metric_entry.get("Dice"),
                    "F1": metric_entry.get("F1"),
                })

    if not rows:
        raise FileNotFoundError("No patient-level Dice/F1 data found for models 700-718")
    return pd.DataFrame(rows)


def load_threshold_patient_rows():
    rows = []
    for threshold, analysis_name in THRESHOLD_ANALYSES.items():
        for model_label, model_name in THRESHOLD_MODELS.items():
            model_dir = os.path.join(ROOT, "full_models", analysis_name, model_name)
            if not os.path.isdir(model_dir):
                continue
            json_pattern = os.path.join(model_dir, f"*{analysis_name}*.json")
            for json_file in sorted(glob.glob(json_pattern)):
                if "ALL_FOLDS" in os.path.basename(json_file):
                    continue
                with open(json_file) as file_handle:
                    data = json.load(file_handle)

                for item in data.get("metric_per_case", []):
                    case_metrics = item.get("metrics")
                    if not isinstance(case_metrics, dict):
                        continue
                    metric_entry = next(iter(case_metrics.values()), None)
                    if not isinstance(metric_entry, dict):
                        continue
                    rows.append({
                        "threshold": threshold,
                        "model_label": model_label,
                        "case_id": extract_case_id(item),
                        "Dice": metric_entry.get("Dice"),
                        "F1": metric_entry.get("F1"),
                    })

    if not rows:
        raise FileNotFoundError("No patient-level threshold data found for the six selected models")
    return pd.DataFrame(rows)


def load_model_results(base_dir, analysis_name, model_type):
    rows = []
    if not os.path.isdir(base_dir):
        return rows
    for model in sorted(os.listdir(base_dir)):
        model_dir = os.path.join(base_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for fold in range(5):
            json_file = os.path.join(model_dir, f"{analysis_name}_fold_{fold}.json")
            if not os.path.exists(json_file):
                continue
            with open(json_file) as file_handle:
                data = json.load(file_handle)
            foreground_mean = data.get("foreground_mean", {})
            rows.append({
                "model_type": model_type,
                "model_name": model,
                "fold": fold,
                "Dice": foreground_mean.get("Dice"),
                "F1": foreground_mean.get("F1"),
                "NSD": foreground_mean.get("NSD"),
            })
    return rows


def compute_feature_importance(analysis_name):
    full_dir = os.path.join(ROOT, "full_models", analysis_name)
    zero_dir = os.path.join(ROOT, "zero_input_models", analysis_name)
    rows = []
    rows.extend(load_model_results(full_dir, analysis_name, "full_model"))
    rows.extend(load_model_results(zero_dir, analysis_name, "zero_input"))

    if not rows:
        return None, None, None

    df = pd.DataFrame(rows).sort_values(["model_type", "model_name", "fold"])
    df_mean = df.groupby(["model_type", "model_name"]).agg(
        Dice_mean=("Dice", "mean"),
        Dice_std=("Dice", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        NSD_mean=("NSD", "mean"),
        NSD_std=("NSD", "std"),
    ).reset_index()
    df_mean = df_mean.sort_values(["model_type", "model_name"])

    importance_rows = []
    for _, row in df_mean.iterrows():
        model_name = row["model_name"]
        if "zero_input_channel" not in model_name:
            continue
        dataset_match = re.search(r"(Dataset\d+)", model_name)
        if not dataset_match:
            continue
        dataset_id = dataset_match.group(1)
        if dataset_id not in BASE_MODELS:
            continue
        full_row = df_mean[
            (df_mean["model_name"] == BASE_MODELS[dataset_id])
            & (df_mean["model_type"] == "full_model")
        ]
        if full_row.empty:
            continue

        full_dice = full_row.iloc[0]["Dice_mean"]
        full_f1 = full_row.iloc[0]["F1_mean"]
        channel_match = re.search(r"zero_input_channel_(.*)", model_name)
        importance_rows.append({
            "channel": channel_match.group(1) if channel_match else model_name,
            "dataset": dataset_id,
            "Dice_drop": full_dice - row["Dice_mean"],
            "F1_drop": full_f1 - row["F1_mean"],
            "full_model_dice": full_dice,
            "zero_input_dice": row["Dice_mean"],
            "full_model_f1": full_f1,
            "zero_input_f1": row["F1_mean"],
        })

    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        importance_df = importance_df.rename(columns={"channel": "zero_input_channel"})
        importance_df = importance_df.sort_values(
            ["dataset", "Dice_drop"],
            ascending=[True, False],
        )
        remaining_columns = [
            column
            for column in importance_df.columns
            if column not in ("dataset", "zero_input_channel")
        ]
        importance_df = importance_df[["dataset", "zero_input_channel"] + remaining_columns]

    return df, df_mean, importance_df


def adjust_color(color, factor):
    """Blend a color with white (factor > 1) or black (factor < 1)."""
    red, green, blue = to_rgb(color)
    if factor >= 1:
        amount = factor - 1
        return tuple(channel + (1 - channel) * amount for channel in (red, green, blue))
    return tuple(channel * factor for channel in (red, green, blue))


def draw_boxplot(ax, values, position, color, rng, width=0.34, alpha=1.0, hatch=None):
    values = pd.to_numeric(values, errors="coerce").dropna().tolist()
    if not values:
        return
    box = ax.boxplot(
        values,
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for artist in box["boxes"]:
        artist.set_facecolor(color)
        artist.set_edgecolor("black")
        artist.set_linewidth(1.2)
        artist.set_alpha(alpha)
        if hatch:
            artist.set_hatch(hatch)
    for artist in box["whiskers"] + box["caps"] + box["medians"]:
        artist.set_color("black")
        artist.set_linewidth(1.2)
    ax.scatter(
        [position],
        [np.mean(values)],
        marker="x",
        color="black",
        s=120,
        zorder=4,
        linewidths=1.8,
    )
    jitter_width = min(0.10, width * 0.35)
    jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))
    ax.scatter(
        np.full(len(values), position) + jitter,
        values,
        color="black",
        alpha=0.35,
        s=12,
        zorder=3,
        linewidths=0,
    )


def load_significance_results():
    if not os.path.exists(STATISTICS_FILE):
        return None
    significance = pd.read_csv(STATISTICS_FILE)
    required_columns = {"metric", "condition_1", "condition_2", "significance"}
    if not required_columns.issubset(significance.columns):
        return None
    return significance[significance["analysis"] == "single_input_models_709_716"].copy()


def load_loo_zero_tests():
    if not os.path.exists(LOO_ZERO_STATISTICS_FILE):
        return None
    return pd.read_csv(LOO_ZERO_STATISTICS_FILE)


def draw_significance_brackets(axis, significance, model_ids, metric, x_offset, y_start):
    model_positions = {
        f"Dataset_{dataset_id}": index + x_offset
        for index, dataset_id in enumerate(model_ids)
    }
    metric_rows = significance[significance["metric"] == metric]
    intervals_by_level = []
    bracket_height = 0.012
    level_spacing = 0.035

    for row in metric_rows.itertuples(index=False):
        first = model_positions.get(row.condition_1)
        second = model_positions.get(row.condition_2)
        if first is None or second is None or row.significance != "ns":
            continue
        left, right = sorted((first, second))
        level = 0
        while level < len(intervals_by_level):
            overlaps = any(
                left < previous_right and right > previous_left
                for previous_left, previous_right in intervals_by_level[level]
            )
            if not overlaps:
                break
            level += 1
        if level == len(intervals_by_level):
            intervals_by_level.append([])
        intervals_by_level[level].append((left, right))

        y = y_start + level * level_spacing
        axis.plot(
            [left, left, right, right],
            [y, y + bracket_height, y + bracket_height, y],
            color="black",
            linewidth=0.8,
            clip_on=False,
        )
        axis.text(
            (left + right) / 2,
            y + bracket_height + 0.004,
            row.significance,
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            clip_on=False,
        )

    if not intervals_by_level:
        return y_start
    return y_start + (len(intervals_by_level) - 1) * level_spacing + bracket_height + 0.025


def save_longi_dice_f1_plot(df, model_ids, output_filename, significance=None, y_limits=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure, axis = plt.subplots(figsize=(15, 6))
    rng = np.random.default_rng(42)
    dice_offset = -0.21
    f1_offset = 0.21

    for index, dataset_id in enumerate(model_ids):
        subset = df[df["dataset_id"] == dataset_id]
        if subset.empty:
            continue
        base_color = GROUP_COLORS[DATASET_GROUP[dataset_id]]
        draw_boxplot(axis, subset["Dice"], index + dice_offset, adjust_color(base_color, 1.45), rng)
        draw_boxplot(axis, subset["F1"], index + f1_offset, adjust_color(base_color, 0.65), rng)

    axis.set_xlabel("Model", fontsize=12)
    axis.set_ylabel("Dice / F1", fontsize=12)
    axis.set_xticks(range(len(model_ids)))
    axis.set_xticklabels([DATASET_LABELS[dataset_id] for dataset_id in model_ids], rotation=45, ha="right", fontsize=10)
    axis.tick_params(axis="y", labelsize=11)
    axis.set_xlim(-0.65, len(model_ids) - 0.35)
    axis.set_ylim(*(y_limits or (-0.02, 1.02)))
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    if significance is not None:
        dice_top = draw_significance_brackets(
            axis, significance, model_ids, "Dice", dice_offset, 1.05
        )
        f1_top = draw_significance_brackets(
            axis, significance, model_ids, "F1", f1_offset, dice_top + 0.04
        )
        axis.set_ylim(-0.02, max(1.02, f1_top + 0.03))

    legend_handles = []
    legend_labels = []
    groups = []
    for dataset_id in model_ids:
        group = DATASET_GROUP[dataset_id]
        if group not in groups:
            groups.append(group)
    for group in groups:
        base_color = GROUP_COLORS[group]
        legend_handles.append((
            Patch(facecolor=adjust_color(base_color, 1.45), edgecolor="black"),
            Patch(facecolor=adjust_color(base_color, 0.65), edgecolor="black"),
        ))
        legend_labels.append(group)
    axis.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
        title="Dataset group\n(light: Dice, dark: F1)",
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=10,
        title_fontsize=10,
        handlelength=2.5,
        handletextpad=0.5,
        labelspacing=0.45,
    )
    figure.tight_layout(rect=[0, 0, 0.86, 1])
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def save_loo_difference_plot(df, zero_tests=None):
    loo_ids = list(range(700, 708))
    ground_truth_id = 708
    selected = df[df["dataset_id"].isin([*loo_ids, ground_truth_id])]
    wide = selected.pivot_table(
        index="case_id",
        columns="dataset_id",
        values=["Dice", "F1"],
        aggfunc="first",
    )

    figure, axis = plt.subplots(figsize=(15, 7.5))
    rng = np.random.default_rng(42)
    dice_offset = -0.21
    f1_offset = 0.21
    significance_y = 0.2
    plotted_values = []
    loo_color = GROUP_COLORS["LOO"]

    for index, dataset_id in enumerate(loo_ids):
        if ("Dice", ground_truth_id) not in wide.columns or ("Dice", dataset_id) not in wide.columns:
            continue
        for metric, offset, color_factor in (
            ("Dice", dice_offset, 1.45),
            ("F1", f1_offset, 0.65),
        ):
            if (metric, dataset_id) not in wide.columns or (metric, ground_truth_id) not in wide.columns:
                continue
            paired = wide[(metric, dataset_id)].dropna().to_frame("loo").join(
                wide[(metric, ground_truth_id)].rename("ground_truth"),
                how="inner",
            )
            differences = paired["loo"] - paired["ground_truth"]
            values = pd.to_numeric(differences, errors="coerce").dropna()
            if values.empty:
                continue
            position = index + offset
            draw_boxplot(axis, values, position, adjust_color(loo_color, color_factor), rng)
            plotted_values.extend(values.tolist())

            if zero_tests is not None:
                match = zero_tests[
                    (zero_tests["metric"] == metric)
                    & (zero_tests["condition"] == f"Dataset_{dataset_id}")
                ]
                if not match.empty and match.iloc[0]["significance"] not in {"ns", "NA"}:
                    significance_label = match.iloc[0]["significance"]
                    axis.annotate(
                        significance_label,
                        xy=(position, significance_y),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                        color="#c62828",
                        clip_on=False,
                    )

    axis.set_xlabel("LOO model", fontsize=12)
    axis.set_ylabel("Score difference (LOO − All together)", fontsize=12)
    axis.set_xticks(range(len(loo_ids)))
    axis.set_xticklabels(
        [DATASET_LABELS[dataset_id] for dataset_id in loo_ids],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    axis.tick_params(axis="y", labelsize=11)
    axis.set_xlim(-0.65, len(loo_ids) - 0.35)
    value_range = max(plotted_values) - min(plotted_values) if plotted_values else 1.0
    lower = min(0.0, min(plotted_values)) if plotted_values else 0.0
    upper = max(0.0, max(plotted_values)) if plotted_values else 1.0
    padding = max(0.01, value_range * 0.01)
    axis.set_ylim(lower - padding, max(upper + padding * 2.5, significance_y + padding * 2))
    axis.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(
        handles=[
            Patch(facecolor=adjust_color(loo_color, 1.45), edgecolor="black", label="Dice"),
            Patch(facecolor=adjust_color(loo_color, 0.65), edgecolor="black", label="F1"),
        ],
        title="Metric",
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=10,
        title_fontsize=10,
    )
    figure.tight_layout(rect=[0, 0, 0.86, 1])
    output_dir = os.path.join(OUTPUT_DIR, "loo_difference_from_gt")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "loo_vs_all_together_Dice_F1_differences.png")
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def save_lesion_size_threshold_plot(df):
    model_order = list(THRESHOLD_MODELS)
    threshold_colors = {
        "all": "#1f77b4",
        "0.3cm": "#ff7f0e",
        "0.5cm": "#2ca02c",
    }
    threshold_labels = {
        "all": "0 cm$^3$ (all lesions)",
        "0.3cm": "0.3 cm$^3$",
        "0.5cm": "0.5 cm$^3$",
    }
    model_spacing = 1.45
    threshold_offsets = (-0.13, 0.0, 0.13)
    metric_offsets = {"Dice": -0.25, "F1": 0.25}
    figure, axis = plt.subplots(figsize=(18, 7))
    rng = np.random.default_rng(42)

    for model_index, model_label in enumerate(model_order):
        model_center = model_index * model_spacing
        model_data = df[df["model_label"] == model_label]

        for metric, metric_offset in metric_offsets.items():
            for threshold_index, threshold in enumerate(THRESHOLD_ORDER):
                subset = model_data[model_data["threshold"] == threshold]
                values = pd.to_numeric(subset[metric], errors="coerce").dropna()
                if values.empty:
                    continue

                draw_boxplot(
                    axis,
                    values,
                    model_center + metric_offset + threshold_offsets[threshold_index],
                    threshold_colors[threshold],
                    rng,
                    width=0.09,
                    alpha=0.95 if metric == "Dice" else 0.5,
                    hatch="//" if metric == "F1" else None,
                )

        axis.axvline(model_center, color="gray", linestyle=":", linewidth=0.8, alpha=0.65)
        for metric, metric_offset in metric_offsets.items():
            axis.text(
                model_center + metric_offset,
                1.035,
                metric,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    model_centers = [index * model_spacing for index in range(len(model_order))]
    axis.set_xticks(model_centers)
    axis.set_xticklabels(
        [label.replace(" (", "\n(") for label in model_order],
        fontsize=9,
    )
    axis.set_xlabel("Model", fontsize=12)
    axis.set_ylabel("Dice / F1", fontsize=12)
    axis.set_title("Dice and F1 by lesion-size threshold", fontsize=14)
    axis.set_xlim(model_centers[0] - 0.62, model_centers[-1] + 0.62)
    axis.set_ylim(0.2, 1.08)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.tick_params(axis="y", labelsize=10)

    threshold_handles = [
        Patch(facecolor=threshold_colors[threshold], edgecolor="black", label=threshold_labels[threshold])
        for threshold in THRESHOLD_ORDER
    ]
    threshold_legend = figure.legend(
        handles=threshold_handles,
        title="Lesion-size\nthreshold",
        loc="upper left",
        bbox_to_anchor=(0.645, 0.94),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    threshold_legend.set_alignment("left")
    metric_handles = [
        Patch(facecolor="#999999", edgecolor="black", label="Dice"),
        Patch(facecolor="#999999", edgecolor="black", hatch="//", label="F1"),
    ]
    figure.legend(
        handles=metric_handles,
        title="Metric",
        loc="upper left",
        bbox_to_anchor=(0.645, 0.55),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )

    axis.set_position([0.08, 0.16, 0.56, 0.76])
    output_path = os.path.join(OUTPUT_DIR, "lesion_size_thresholds_Dice_F1_boxplot.png")
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def save_iou_threshold_f1_plot():
    if not os.path.exists(IOU_F1_SUMMARY_FILE):
        print(f"IoU threshold summary not found: {IOU_F1_SUMMARY_FILE}")
        return

    df = pd.read_csv(IOU_F1_SUMMARY_FILE)
    required_columns = {"model", "iou_threshold", "F1_mean", "F1_std"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        print(f"Missing columns in IoU threshold summary: {sorted(missing_columns)}")
        return

    output_dir = os.path.join(OUTPUT_DIR, "iou_threshold_analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "iou_threshold_f1_plot.png")

    model_labels = {
        "Dataset708_MM_Lesion_seg_all_together": "All together",
        "Dataset709_MM_Lesion_seg_just_ConvCT": "ConvCT",
        "Dataset710_MM_Lesion_seg_just_VMI_40": "VMI40",
        "Dataset713_MM_Lesion_seg_just_CaSupp_25": "CaSupp25",
        "Dataset717_MM_Lesion_seg_all_VMI": "All VMI",
        "Dataset718_MM_Lesion_seg_all_CaSupp": "All CaSupp",
    }
    model_colors = {
        "Dataset709_MM_Lesion_seg_just_ConvCT": "#1f77b4",
        "Dataset710_MM_Lesion_seg_just_VMI_40": "#ff7f0e",
        "Dataset713_MM_Lesion_seg_just_CaSupp_25": "#2ca02c",
        "Dataset708_MM_Lesion_seg_all_together": "#d62728",
        "Dataset717_MM_Lesion_seg_all_VMI": "#9467bd",
        "Dataset718_MM_Lesion_seg_all_CaSupp": "#8c564b",
    }

    figure, axis = plt.subplots(figsize=(12, 5))
    for model, model_df in df.groupby("model", sort=False):
        model_df = model_df.sort_values("iou_threshold")
        axis.errorbar(
            model_df["iou_threshold"],
            model_df["F1_mean"],
            yerr=model_df["F1_std"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=3,
            color=model_colors.get(model, "#4C78A8"),
            label=model_labels.get(model, model),
        )

    axis.set_title("F1 score versus IoU threshold")
    axis.set_xlabel("IoU threshold")
    axis.set_ylabel("F1")
    axis.set_xticks(sorted(df["iou_threshold"].unique()))
    axis.set_ylim(0, 1)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(loc="best", frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def save_feature_importance_outputs(threshold_label, df, df_mean, importance_df):
    threshold_dir = os.path.join(OUTPUT_DIR, "feature_importance", threshold_label)
    images_dir = os.path.join(threshold_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    results_csv = os.path.join(threshold_dir, "results_all_folds.csv")
    mean_csv = os.path.join(threshold_dir, "results_model_means.csv")
    importance_csv = os.path.join(threshold_dir, "feature_importance.csv")
    summary_xlsx = os.path.join(threshold_dir, "feature_importance_summary.xlsx")

    df.to_csv(results_csv, index=False)
    df_mean.to_csv(mean_csv, index=False)
    importance_df.to_csv(importance_csv, index=False)
    print(f"Saved feature importance tables for {threshold_label}: {threshold_dir}")

    if importance_df.empty:
        with pd.ExcelWriter(summary_xlsx) as writer:
            df.to_excel(writer, sheet_name="all_folds", index=False)
            df_mean.to_excel(writer, sheet_name="model_means", index=False)
            importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        return

    for dataset in importance_df["dataset"].unique():
        subset = importance_df[importance_df["dataset"] == dataset].sort_values(
            "Dice_drop",
            ascending=False,
        )
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(subset["zero_input_channel"], subset["Dice_drop"])
        axis.set_ylabel("Dice drop (importance)")
        axis.set_xlabel("Removed channel")
        axis.set_title(f"Channel importance ({threshold_label} — {dataset})")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        plot_path = os.path.join(
            images_dir,
            f"feature_importance_{threshold_label}_{dataset}.png",
        )
        figure.savefig(plot_path, dpi=600, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved: {plot_path}")

    label_map = {
        "Dataset708": "All together",
        "Dataset717": "All VMI",
        "Dataset718": "All CaSupp",
    }
    dice_heatmap_df = importance_df.pivot_table(
        index="zero_input_channel",
        columns="dataset",
        values="Dice_drop",
    )
    f1_heatmap_df = importance_df.pivot_table(
        index="zero_input_channel",
        columns="dataset",
        values="F1_drop",
    )

    channel_patterns = [
        (r"conv\s*[_-]?\s*ct", 0),
        (r"vmi\s*[_-]?\s*40", 1),
        (r"vmi\s*[_-]?\s*80", 2),
        (r"vmi\s*[_-]?\s*120", 3),
        (r"ca\s*[_-]?\s*supp\s*[_-]?\s*25", 4),
        (r"ca\s*[_-]?\s*supp\s*[_-]?\s*50", 5),
        (r"ca\s*[_-]?\s*supp\s*[_-]?\s*75", 6),
        (r"ca\s*[_-]?\s*supp\s*[_-]?\s*100", 7),
    ]

    def channel_order(channel):
        channel_text = str(channel).lower()
        for pattern, order in channel_patterns:
            if re.search(pattern, channel_text):
                return order
        return len(channel_patterns)

    ordered_channels = sorted(dice_heatmap_df.index, key=channel_order, reverse=True)
    dice_heatmap_df = dice_heatmap_df.reindex(ordered_channels).rename(columns=label_map)
    f1_heatmap_df = f1_heatmap_df.reindex(ordered_channels).rename(columns=label_map)

    dice_heatmap_df.to_csv(os.path.join(threshold_dir, "feature_importance_heatmap_Dice.csv"))
    f1_heatmap_df.to_csv(os.path.join(threshold_dir, "feature_importance_heatmap_F1.csv"))

    with pd.ExcelWriter(summary_xlsx) as writer:
        df.to_excel(writer, sheet_name="all_folds", index=False)
        df_mean.to_excel(writer, sheet_name="model_means", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        dice_heatmap_df.to_excel(writer, sheet_name="heatmap_Dice")
        f1_heatmap_df.to_excel(writer, sheet_name="heatmap_F1")

    dice_cbar_min = dice_heatmap_df.min().min()
    dice_cbar_max = dice_heatmap_df.max().max()
    for metric, heatmap_df, cbar_min, cbar_max in (
        ("Dice", dice_heatmap_df, dice_cbar_min, dice_cbar_max),
        ("F1", f1_heatmap_df, dice_cbar_min, dice_cbar_max),
    ):
        figure, axis = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap="viridis",
            fmt=".3f",
            linewidths=0.5,
            vmin=cbar_min,
            vmax=cbar_max,
            cbar_kws={"label": f"{metric} drop"},
            ax=axis,
        )
        axis.set_title(f"Channel importance heatmap — {metric} drop")
        axis.set_ylabel("Zeroed channel")
        axis.set_xlabel("Dataset")
        figure.tight_layout()
        heatmap_path = os.path.join(
            images_dir,
            f"feature_importance_heatmap_{threshold_label}_{metric}.png",
        )
        figure.savefig(heatmap_path, dpi=600, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved: {heatmap_path}")


def main():
    for threshold_label, analysis_name in THRESHOLD_ANALYSES.items():
        df, df_mean, importance_df = compute_feature_importance(analysis_name)
        if df is None:
            print(f"No feature importance results for threshold: {threshold_label}")
            continue
        save_feature_importance_outputs(
            threshold_label,
            df,
            df_mean,
            importance_df,
        )

    save_iou_threshold_f1_plot()

    patient_df = load_patient_rows()
    significance = load_significance_results()
    loo_zero_tests = load_loo_zero_tests()
    save_longi_dice_f1_plot(
        patient_df,
        list(range(700, 719)),
        "models_700_718_Dice_F1_boxplot.png",
    )
    save_longi_dice_f1_plot(
        patient_df,
        list(range(709, 717)),
        "single_input_709_716_Dice_F1_boxplot.png",
        significance=significance,
    )
    save_longi_dice_f1_plot(
        patient_df,
        list(range(700, 709)),
        "leave_one_out_700_708_Dice_F1_boxplot.png",
    )
    save_longi_dice_f1_plot(
        patient_df,
        [708, 717, 718],
        "grouped_708_717_718_Dice_F1_boxplot.png",
        y_limits=(0.4, 1.0),
    )
    save_loo_difference_plot(patient_df, loo_zero_tests)
    threshold_df = load_threshold_patient_rows()
    save_lesion_size_threshold_plot(threshold_df)


if __name__ == "__main__":
    main()
