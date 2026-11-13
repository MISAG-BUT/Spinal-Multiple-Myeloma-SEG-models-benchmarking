"""Generate publication figures from stored numerical benchmarking results."""

import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


plt.rcParams.update({
    "font.size": 19,
    "font.weight": "bold",
    "axes.titlesize": 23,
    "axes.titleweight": "bold",
    "axes.labelsize": 21,
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.2,
    "legend.fontsize": 18,
    "legend.title_fontsize": 19,
})


ANALYSIS = "longi_summary_all"
JSON_ROOT = os.path.join(config.FULL_MODELS_DIR, ANALYSIS)
OUTPUT_DIR = config.FIGURES_DIR
STATISTICS_FILE = os.path.join(
    config.TABLES_DIR,
    "statistical_analysis",
    "single_input_pairwise_tests.csv",
)
LOO_ZERO_STATISTICS_FILE = os.path.join(
    config.TABLES_DIR,
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
    config.TABLES_DIR,
    "iou_threshold_analysis",
    "iou_threshold_f1_summary.csv",
)


def save_figure(figure, output_path):
    output_stem, _ = os.path.splitext(output_path)
    for extension in ("png", "eps", "svg"):
        figure_path = f"{output_stem}.{extension}"
        figure.savefig(
            figure_path,
            format=extension,
            dpi=600,
            bbox_inches="tight",
        )
        print(f"Saved: {figure_path}")


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
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "dataset_label": f"Dataset_{dataset_id}",
                    "case_id": extract_case_id(item)
                    or f"{os.path.basename(json_file)}_{len(rows)}",
                    "Dice": metric_entry.get("Dice"),
                    "F1": metric_entry.get("F1"),
                    "NSD": metric_entry.get("NSD"),
                })

    if not rows:
        raise FileNotFoundError("No patient-level Dice/F1 data found for models 700-718")
    return pd.DataFrame(rows)


def load_threshold_patient_rows():
    rows = []
    for threshold, analysis_name in THRESHOLD_ANALYSES.items():
        for model_label, model_name in THRESHOLD_MODELS.items():
            model_dir = os.path.join(config.FULL_MODELS_DIR, analysis_name, model_name)
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
                        "model_name": model_name,
                        "case_id": extract_case_id(item)
                        or f"{os.path.basename(json_file)}_{len(rows)}",
                        "Dice": metric_entry.get("Dice"),
                        "F1": metric_entry.get("F1"),
                        "NSD": metric_entry.get("NSD"),
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
    full_dir = os.path.join(config.FULL_MODELS_DIR, analysis_name)
    zero_dir = os.path.join(config.ZERO_INPUT_MODELS_DIR, analysis_name)
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
        artist.set_linewidth(2.0)
        artist.set_alpha(alpha)
        if hatch:
            artist.set_hatch(hatch)
    for artist in box["whiskers"] + box["caps"] + box["medians"]:
        artist.set_color("black")
        artist.set_linewidth(2.0)
    ax.scatter(
        [position],
        [np.mean(values)],
        marker="x",
        color="black",
        s=150,
        zorder=4,
        linewidths=2.4,
    )
    jitter_width = min(0.10, width * 0.35)
    jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))
    ax.scatter(
        np.full(len(values), position) + jitter,
        values,
        color="black",
        alpha=0.45,
        s=20,
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
            linewidth=1.8,
            clip_on=False,
        )
        axis.text(
            (left + right) / 2,
            y + bracket_height + 0.004,
            row.significance,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

    if not intervals_by_level:
        return y_start
    return y_start + (len(intervals_by_level) - 1) * level_spacing + bracket_height + 0.025


def fit_horizontal_labels(figure, axis):
    """Keep horizontal labels when they fit; otherwise use the original tilt."""
    figure.canvas.draw()
    labels = axis.get_xticklabels()
    boxes = [label.get_window_extent() for label in labels]
    if any(left.x1 + 8 > right.x0 for left, right in zip(boxes, boxes[1:])):
        for label in labels:
            label.set_rotation(45)
            label.set_ha("right")
        figure.tight_layout()


def save_longi_dice_f1_plot(df, model_ids, output_filename, significance=None, y_limits=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure, axis = plt.subplots(figsize=(21, 8))
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

    axis.set_xlabel("Model", fontsize=21, fontweight="bold")
    axis.set_ylabel("Dice / F1", fontsize=21, fontweight="bold")
    axis.set_xticks(range(len(model_ids)))
    horizontal_labels = output_filename.startswith("grouped_")
    axis.set_xticklabels([DATASET_LABELS[dataset_id].replace("LOO w/o ", "LOO w/o\n") if horizontal_labels else DATASET_LABELS[dataset_id] for dataset_id in model_ids], rotation=0 if horizontal_labels else 45, ha="center" if horizontal_labels else "right", fontsize=18, fontweight="bold")
    axis.tick_params(axis="y", labelsize=18, width=1.6)
    axis.set_xlim(-0.65, len(model_ids) - 0.35)
    axis.set_ylim(*(y_limits or (-0.02, 1.02)))
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    if significance is not None:
        marker_shift = 0.03 if output_filename.startswith("single_input_") else 0.0
        dice_top = draw_significance_brackets(
            axis, significance, model_ids, "Dice", dice_offset, 1.05 - marker_shift
        )
        f1_top = draw_significance_brackets(
            axis, significance, model_ids, "F1", f1_offset, dice_top + 0.04
        )
        axis.set_ylim(-0.02, max(1.02, f1_top + 0.03 + marker_shift))

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
    if output_filename.startswith("single_input_"):
        legend_location, legend_anchor, legend_columns = "lower left", (-0.015, 0.005), len(groups)
    elif output_filename.startswith("models_"):
        legend_location, legend_anchor, legend_columns = "lower left", (0.01, 0.02), 3
    else:
        legend_location, legend_anchor, legend_columns = "lower center", (0.5, 0.07), len(groups)
    legend_title = "Dataset group (light: Dice, dark: F1)"
    if output_filename.startswith(("grouped_", "leave_one_out_")):
        legend_handles = [Patch(visible=False)] + legend_handles
        legend_labels = [legend_title + ":"] + legend_labels
        legend_title = None
        legend_location, legend_anchor, legend_columns = "lower left", (0.01, 0.07), len(legend_handles)
        if output_filename.startswith(("grouped_", "leave_one_out_")):
            legend_anchor = (-0.015, 0.0)
        if output_filename.startswith("grouped_"):
            legend_anchor = (-0.015, 0.02)
    legend_spacing = {}
    if output_filename.startswith("single_input_"):
        legend_handles = [Patch(visible=False)] + legend_handles
        legend_labels = [legend_title + ":"] + legend_labels
        legend_title = None
        legend_columns = len(legend_handles)
        legend_spacing = dict(borderaxespad=0, columnspacing=0.6)
    if output_filename.startswith("models_"):
        legend_columns = len(legend_handles)
        legend_anchor = (0.02, 0.005)
        legend_spacing = dict(borderaxespad=0, columnspacing=0.6)
    if output_filename.startswith(("grouped_", "leave_one_out_")):
        legend_spacing = dict(borderaxespad=0, columnspacing=0.6)
    legend = axis.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
        title=legend_title,
        loc=legend_location,
        bbox_to_anchor=legend_anchor,
        ncol=legend_columns,
        frameon=False,
        fontsize=20 if output_filename.startswith(("grouped_", "leave_one_out_")) else 18,
        title_fontsize=19,
        handlelength=1.8 if output_filename.startswith(("grouped_", "leave_one_out_")) else 2.5,
        handletextpad=0.3 if output_filename.startswith(("grouped_", "leave_one_out_")) else 0.5,
        labelspacing=0.45,
        **legend_spacing,
    )
    legend.set_alignment("left")
    figure.tight_layout()
    if horizontal_labels:
        fit_horizontal_labels(figure, axis)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    save_figure(figure, output_path)
    plt.close(figure)


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

    figure, axis = plt.subplots(figsize=(17, 8.5))
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
                        fontsize=17,
                        fontweight="bold",
                        color="#c62828",
                        clip_on=False,
                    )

    axis.set_xlabel("LOO model", fontsize=21, fontweight="bold")
    axis.set_ylabel("Score difference (LOO − All together)", fontsize=21, fontweight="bold")
    axis.set_xticks(range(len(loo_ids)))
    axis.set_xticklabels(
        [DATASET_LABELS[dataset_id] for dataset_id in loo_ids],
        rotation=45,
        ha="right",
        fontsize=18,
        fontweight="bold",
    )
    axis.tick_params(axis="y", labelsize=18, width=1.6)
    axis.set_xlim(-0.65, len(loo_ids) - 0.35)
    value_range = max(plotted_values) - min(plotted_values) if plotted_values else 1.0
    lower = min(0.0, min(plotted_values)) if plotted_values else 0.0
    upper = max(0.0, max(plotted_values)) if plotted_values else 1.0
    padding = max(0.01, value_range * 0.01)
    axis.set_ylim(lower - padding, max(upper + padding * 2.5, significance_y + padding * 2))
    axis.axhline(y=0, color="gray", linestyle="--", linewidth=2, alpha=0.8)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(
        handles=[
            Patch(visible=False, label="Metric:"),
            Patch(facecolor=adjust_color(loo_color, 1.45), edgecolor="black", label="Dice"),
            Patch(facecolor=adjust_color(loo_color, 0.65), edgecolor="black", label="F1"),
        ],
        loc="upper right",
        ncol=3,
        columnspacing=0.6,
        frameon=False,
        fontsize=18,
        title_fontsize=19,
    )
    figure.tight_layout()
    output_dir = os.path.join(OUTPUT_DIR, "loo_difference_from_gt")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "loo_vs_all_together_Dice_F1_differences.png")
    save_figure(figure, output_path)
    plt.close(figure)


def save_lesion_size_threshold_plot(df):
    model_order = list(THRESHOLD_MODELS)
    threshold_colors = {
        "all": "#1f77b4",
        "0.3cm": "#ff7f0e",
        "0.5cm": "#2ca02c",
    }
    threshold_labels = {
        "all": "0 cm³ (all lesions)",
        "0.3cm": "0.3 cm³",
        "0.5cm": "0.5 cm³",
    }
    model_spacing = 1.45
    threshold_offsets = (-0.13, 0.0, 0.13)
    metric_offsets = {"Dice": -0.25, "F1": 0.25}
    figure, axis = plt.subplots(figsize=(20, 8))
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

        axis.axvline(model_center, color="gray", linestyle=":", linewidth=1.5, alpha=0.75)
        for metric, metric_offset in metric_offsets.items():
            axis.text(
                model_center + metric_offset,
                1.035,
                metric,
                ha="center",
                va="bottom",
                fontsize=19,
                fontweight="bold",
            )

    model_centers = [index * model_spacing for index in range(len(model_order))]
    axis.set_xticks(model_centers)
    axis.set_xticklabels(
        [label.replace(" (", "\n(") for label in model_order],
        fontsize=17,
        fontweight="bold",
    )
    axis.set_xlabel("Model", fontsize=21, fontweight="bold")
    axis.set_ylabel("Dice / F1", fontsize=21, fontweight="bold")
    axis.set_xlim(model_centers[0] - 0.62, model_centers[-1] + 0.62)
    axis.set_ylim(0.01, 1.1)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.tick_params(axis="y", labelsize=18, width=1.6)

    threshold_handles = [
        Patch(facecolor=threshold_colors[threshold], edgecolor="black", label=threshold_labels[threshold])
        for threshold in THRESHOLD_ORDER
    ]
    legend = axis.legend(
        handles=[Patch(visible=False)] + threshold_handles + [
            Patch(facecolor="#999999", edgecolor="black", label="Dice"),
            Patch(facecolor="#999999", edgecolor="black", hatch="//", label="F1"),
        ],
        labels=["Lesion-size threshold:"]
        + [threshold_labels[threshold] for threshold in THRESHOLD_ORDER]
        + ["Dice", "F1"],
        loc="lower left",
        bbox_to_anchor=(0.01, 0.005),
        ncol=6,
        frameon=False,
        fontsize=17,
        borderaxespad=0,
        columnspacing=0.6,
        handlelength=1.8,
        handletextpad=0.3,
    )
    legend.set_alignment("left")

    figure.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "lesion_size_thresholds_Dice_F1_boxplot.png")
    save_figure(figure, output_path)
    plt.close(figure)


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

    figure, axis = plt.subplots(figsize=(14, 7))
    for model, model_df in df.groupby("model", sort=False):
        model_df = model_df.sort_values("iou_threshold")
        axis.errorbar(
            model_df["iou_threshold"],
            model_df["F1_mean"],
            yerr=model_df["F1_std"],
            marker="o",
            markersize=7,
            linewidth=2.5,
            capsize=5,
            color=model_colors.get(model, "#4C78A8"),
            label=model_labels.get(model, model),
        )

    axis.set_xlabel("IoU threshold", fontsize=21, fontweight="bold")
    axis.set_ylabel("F1", fontsize=21, fontweight="bold")
    axis.set_xticks(np.arange(0.1, 1.0, 0.1))
    axis.set_ylim(0, 1)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    model_label = axis.text(
        0.02, 0.95, "Model:", transform=axis.transAxes,
        fontsize=18, fontweight="bold", ha="left", va="center",
    )
    figure.canvas.draw()
    label_box = model_label.get_window_extent()
    legend_x = axis.transAxes.inverted().transform((label_box.x1 + 12, label_box.y0))[0]
    legend = axis.legend(
        loc="upper left", bbox_to_anchor=(legend_x, 0.98),
        ncol=3, frameon=False, borderaxespad=0,
        columnspacing=1.2, handletextpad=0.5,
    )
    legend.set_alignment("left")
    figure.canvas.draw()
    legend_box = legend.get_window_extent()
    model_label.set_y(axis.transAxes.inverted().transform(
        (legend_box.x0, (legend_box.y0 + legend_box.y1) / 2)
    )[1])
    save_figure(figure, output_path)
    plt.close(figure)


def save_feature_importance_plots(threshold_label, importance_df):
    if importance_df.empty:
        return

    images_dir = os.path.join(OUTPUT_DIR, "feature_importance", threshold_label)
    os.makedirs(images_dir, exist_ok=True)
    for dataset in importance_df["dataset"].unique():
        subset = importance_df[importance_df["dataset"] == dataset].sort_values(
            "Dice_drop",
            ascending=False,
        )
        figure, axis = plt.subplots(figsize=(10, 7))
        axis.bar(
            subset["zero_input_channel"],
            subset["Dice_drop"],
            edgecolor="black",
            linewidth=1.5,
        )
        axis.set_ylabel("Dice drop (importance)", fontsize=21, fontweight="bold")
        axis.set_xlabel("Removed channel", fontsize=21, fontweight="bold")
        axis.set_title(f"Channel importance ({threshold_label} — {dataset})", fontsize=23, fontweight="bold")
        axis.tick_params(axis="x", rotation=45, width=1.6)
        figure.tight_layout()
        plot_path = os.path.join(
            images_dir,
            f"feature_importance_{threshold_label}_{dataset}.png",
        )
        save_figure(figure, plot_path)
        plt.close(figure)

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

    dice_cbar_min = dice_heatmap_df.min().min()
    dice_cbar_max = dice_heatmap_df.max().max()
    for metric, heatmap_df, cbar_min, cbar_max in (
        ("Dice", dice_heatmap_df, dice_cbar_min, dice_cbar_max),
        ("F1", f1_heatmap_df, dice_cbar_min, dice_cbar_max),
    ):
        figure, axis = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap="viridis",
            fmt=".3f",
            linewidths=1.2,
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            vmin=cbar_min,
            vmax=cbar_max,
            cbar_kws={"label": f"{metric} drop"},
            ax=axis,
        )
        axis.set_title(f"Channel importance heatmap — {metric} drop", fontsize=23, fontweight="bold")
        axis.set_ylabel("Zeroed channel", fontsize=21, fontweight="bold")
        axis.set_xlabel("Dataset", fontsize=21, fontweight="bold")
        if threshold_label == "all":
            axis.set_title(f"{metric} drop", fontsize=18, fontweight="normal")
            axis.set_ylabel("Zeroed channel", fontsize=16, fontweight="normal")
            axis.set_xlabel("Dataset", fontsize=16, fontweight="normal")
            axis.tick_params(axis="both", labelsize=14)
            axis.set_xticklabels(axis.get_xticklabels(), rotation=0, ha="center")
            colorbar = axis.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=14)
            colorbar.set_label(f"{metric} drop", fontsize=16, fontweight="normal")
            for label in axis.get_xticklabels() + axis.get_yticklabels() + colorbar.ax.get_yticklabels():
                label.set_fontweight("normal")
        figure.tight_layout()
        heatmap_path = os.path.join(
            images_dir,
            f"feature_importance_heatmap_{threshold_label}_{metric}.png",
        )
        save_figure(figure, heatmap_path)
        plt.close(figure)


def main():
    for threshold_label, analysis_name in THRESHOLD_ANALYSES.items():
        _, _, importance_df = compute_feature_importance(analysis_name)
        if importance_df is None:
            print(f"No feature importance results for threshold: {threshold_label}")
            continue
        save_feature_importance_plots(threshold_label, importance_df)

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
        y_limits=(0.3, 1.0),
    )
    save_longi_dice_f1_plot(
        patient_df,
        [708, 717, 718],
        "grouped_708_717_718_Dice_F1_boxplot.png",
        y_limits=(0.3, 1.0),
    )
    save_loo_difference_plot(patient_df, loo_zero_tests)
    threshold_df = load_threshold_patient_rows()
    save_lesion_size_threshold_plot(threshold_df)


if __name__ == "__main__":
    main()
