import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


TRAINING_RESULTS_ROOT = config.NNUNET_RESULTS
GPU_LOG_DIR = config.GPU_LOG_DIR
TIME_LOG = os.path.join(config.INFERENCE_LOG_DIR, "inference_time_log.csv")


def _numeric_column(values, unit):
    return pd.to_numeric(values.astype(str).str.replace(unit, "", regex=False).str.strip())


def build_gpu_tables():
    time_df = pd.read_csv(TIME_LOG).rename(columns={
        "MODEL": "model",
        "FOLD": "fold",
        "DURATION_SEC": "runtime_sec",
        "PARAMETERS": "parameters",
    })
    required_runtime_columns = {"model", "fold", "runtime_sec", "parameters"}
    missing_runtime_columns = required_runtime_columns.difference(time_df.columns)
    if missing_runtime_columns:
        raise ValueError(f"Missing runtime log columns: {sorted(missing_runtime_columns)}")
    if not os.path.isdir(GPU_LOG_DIR):
        raise FileNotFoundError(f"Missing GPU log directory: {GPU_LOG_DIR}")

    gpu_rows = []
    for filename in sorted(os.listdir(GPU_LOG_DIR)):
        if not filename.endswith(".csv"):
            continue
        model, fold_text = filename[:-4].split("_fold_")
        gpu_log = pd.read_csv(os.path.join(GPU_LOG_DIR, filename), skipinitialspace=True)
        required_gpu_columns = {
            "utilization.gpu [%]",
            "utilization.memory [%]",
            "memory.used [MiB]",
            "power.draw [W]",
        }
        missing_gpu_columns = required_gpu_columns.difference(gpu_log.columns)
        if missing_gpu_columns:
            raise ValueError(
                f"Missing GPU log columns in {filename}: {sorted(missing_gpu_columns)}"
            )

        gpu_utilization = _numeric_column(gpu_log["utilization.gpu [%]"], "%")
        memory_used = _numeric_column(gpu_log["memory.used [MiB]"], "MiB")
        power_draw = _numeric_column(gpu_log["power.draw [W]"], "W")
        gpu_rows.append({
            "model": model,
            "fold": int(fold_text),
            "avg_gpu_util_percent": gpu_utilization.mean(),
            "max_gpu_util_percent": gpu_utilization.max(),
            "avg_memory_MiB": memory_used.mean(),
            "max_memory_MiB": memory_used.max(),
            "avg_power_W": power_draw.mean(),
            "energy_Wh": power_draw.sum() / 3600,
        })

    gpu_df = pd.DataFrame(gpu_rows)
    merged = gpu_df.merge(
        time_df[["model", "fold", "runtime_sec", "parameters"]],
        on=["model", "fold"],
        how="left",
    ).sort_values(["model", "fold"]).reset_index(drop=True)

    model_summary = merged.groupby("model").agg(
        runtime_sec=("runtime_sec", "mean"),
        parameters=("parameters", "mean"),
        avg_gpu_util_percent=("avg_gpu_util_percent", "mean"),
        max_gpu_util_percent=("max_gpu_util_percent", "max"),
        avg_memory_MiB=("avg_memory_MiB", "mean"),
        max_memory_MiB=("max_memory_MiB", "max"),
        avg_power_W=("avg_power_W", "mean"),
        energy_Wh=("energy_Wh", "sum"),
    ).reset_index().sort_values("model").reset_index(drop=True)

    article_metrics = [
        "runtime_sec",
        "avg_gpu_util_percent",
        "max_gpu_util_percent",
        "avg_memory_MiB",
        "max_memory_MiB",
        "avg_power_W",
        "energy_Wh",
    ]
    article_groups = {
        "LOO (700-707)": set(range(700, 708)),
        "All together (708)": {708},
        "Single input (709-716)": set(range(709, 717)),
        "All VMI (717)": {717},
        "All CaSupp (718)": {718},
    }
    article_source = merged[
        ~merged["model"].astype(str).str.contains("zero.?input", case=False, regex=True, na=False)
    ].copy()
    article_source["dataset_id"] = pd.to_numeric(
        article_source["model"].str.extract(r"Dataset(\d+)")[0],
        errors="coerce",
    )
    article_summary_rows = []
    for group_name, dataset_ids in article_groups.items():
        group_data = article_source[article_source["dataset_id"].isin(dataset_ids)]
        if group_data.empty:
            continue
        summary_row = {"model_group": group_name}
        parameter_values = pd.to_numeric(group_data["parameters"], errors="coerce").dropna()
        summary_row["parameters"] = (
            parameter_values.median() if not parameter_values.empty else None
        )
        for metric in article_metrics:
            values = pd.to_numeric(group_data[metric], errors="coerce").dropna()
            summary_row[f"{metric}_median"] = values.median() if not values.empty else None
            summary_row[f"{metric}_range_min"] = values.min() if not values.empty else None
            summary_row[f"{metric}_range_max"] = values.max() if not values.empty else None
        article_summary_rows.append(summary_row)
    article_summary = pd.DataFrame(article_summary_rows).round(3)

    training_gpu_rows = []
    for dataset_id in range(700, 719):
        model_pattern = os.path.join(
            TRAINING_RESULTS_ROOT,
            f"Dataset{dataset_id}_MM_Lesion_seg_*",
            "nnUNetTrainer__nnUNetPlans__3d_fullres",
            "fold_*",
            "debug.json",
        )
        for debug_file in sorted(glob.glob(model_pattern)):
            fold_match = re.search(r"/fold_(\d+)/debug\.json$", debug_file)
            if not fold_match:
                continue
            fold = int(fold_match.group(1))
            if fold not in range(5):
                continue
            fold_dir = os.path.dirname(debug_file)
            model_dir = os.path.dirname(os.path.dirname(fold_dir))
            with open(debug_file) as file_handle:
                debug_data = json.load(file_handle)

            epoch_times = []
            for training_log in sorted(glob.glob(os.path.join(fold_dir, "training_log_*.txt"))):
                with open(training_log) as file_handle:
                    epoch_times.extend(
                        float(value)
                        for value in re.findall(
                            r"Epoch time:\s*([0-9]+(?:\.[0-9]+)?)\s*s",
                            file_handle.read(),
                        )
                    )
            training_gpu_rows.append({
                "dataset_id": dataset_id,
                "model": os.path.basename(model_dir),
                "fold": fold,
                "gpu_name": debug_data.get("gpu_name"),
                "mean_epoch_time_sec": (
                    sum(epoch_times) / len(epoch_times) if epoch_times else None
                ),
                "epoch_count": len(epoch_times),
            })

    training_gpu_df = pd.DataFrame(training_gpu_rows).sort_values(
        ["dataset_id", "model", "fold"]
    ).reset_index(drop=True)
    training_gpu_summary = training_gpu_df.groupby(
        ["dataset_id", "model", "gpu_name"],
        dropna=False,
    ).agg(
        folds=("fold", "nunique"),
        mean_epoch_time_sec=("mean_epoch_time_sec", "mean"),
        epoch_count=("epoch_count", "sum"),
    ).reset_index().sort_values(
        ["dataset_id", "model", "gpu_name"]
    ).reset_index(drop=True)

    return {
        "gpu_usage_summary.csv": merged,
        "gpu_usage_model_means.csv": model_summary,
        "gpu_usage_article_summary.csv": article_summary,
        "training_gpu_usage.csv": training_gpu_df,
        "training_gpu_usage_summary.csv": training_gpu_summary,
    }


if __name__ == "__main__":
    from generate_suplementary_tables import main

    main()
