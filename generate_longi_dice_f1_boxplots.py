import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def adjust_color(color, factor):
    """Blend a color with white (factor > 1) or black (factor < 1)."""
    red, green, blue = to_rgb(color)
    if factor >= 1:
        amount = factor - 1
        return tuple(channel + (1 - channel) * amount for channel in (red, green, blue))
    return tuple(channel * factor for channel in (red, green, blue))


def draw_boxplot(ax, values, position, color, rng):
    values = pd.to_numeric(values, errors="coerce").dropna().tolist()
    if not values:
        return
    box = ax.boxplot(
        values,
        positions=[position],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for artist in box["boxes"]:
        artist.set_facecolor(color)
        artist.set_edgecolor("black")
        artist.set_linewidth(1.2)
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
    jitter = rng.uniform(-0.10, 0.10, size=len(values))
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


def save_longi_dice_f1_plot(df, model_ids, output_filename, significance=None):
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
    axis.set_ylim(-0.02, 1.02)
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


def main():
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
    )
    save_loo_difference_plot(patient_df, loo_zero_tests)


if __name__ == "__main__":
    main()
