import glob
import json
import os
import re

import pandas as pd


ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
TRAINING_RESULTS_ROOT = "/home/nohel/DATA/nnUNet_results"
TIME_LOG = os.path.join(ROOT, "inference_logs", "inference_time_log.csv")
OUTPUT_DIR = os.path.join(ROOT, "results_final", "tables_final", "model_configuration")
CONFIGURATION = "3d_fullres"
TRAINER_DIR = f"nnUNetTrainer__nnUNetPlans__{CONFIGURATION}"
DATASET_IDS = range(700, 719)


def read_json(path):
    with open(path) as file_handle:
        return json.load(file_handle)


def short_name(class_path):
    return class_path.split(".")[-1] if class_path else None


def read_training_settings(trainer_dir):
    settings = []
    for debug_file in sorted(glob.glob(os.path.join(trainer_dir, "fold_*", "debug.json"))):
        debug = read_json(debug_file)
        loss_match = re.search(r"\((loss)\): (\w+)", debug.get("loss", ""))
        optimizer_text = debug.get("optimizer", "")

        def optimizer_value(name):
            match = re.search(rf"{name}: (\S+)", optimizer_text)
            return match.group(1) if match else None

        settings.append({
            "loss_function": loss_match.group(2) if loss_match else None,
            "deep_supervision": debug.get("enable_deep_supervision"),
            "optimizer": optimizer_text.split()[0] if optimizer_text else None,
            "initial_lr": debug.get("initial_lr"),
            "momentum": optimizer_value("momentum"),
            "nesterov": optimizer_value("nesterov"),
            "weight_decay": debug.get("weight_decay"),
            "lr_scheduler": short_name(
                re.sub(r"^<(.*) object at .*>$", r"\1", debug.get("lr_scheduler", ""))
            ),
            "num_epochs": debug.get("num_epochs"),
            "iterations_per_epoch": debug.get("num_iterations_per_epoch"),
            "oversample_foreground_percent": debug.get("oversample_foreground_percent"),
            "mirroring_axes": debug.get("inference_allowed_mirroring_axes"),
            "gpu_name": debug.get("gpu_name"),
            "torch_version": debug.get("torch_version"),
        })
    return settings


def format_value(value):
    if isinstance(value, list):
        return str(value).replace(" ", "")
    return value


def build_configuration_rows():
    parameters = {}
    if os.path.exists(TIME_LOG):
        time_df = pd.read_csv(TIME_LOG)
        parameters = time_df.groupby("MODEL")["PARAMETERS"].first().to_dict()

    rows = []
    for dataset_id in DATASET_IDS:
        trainer_dirs = glob.glob(
            os.path.join(TRAINING_RESULTS_ROOT, f"Dataset{dataset_id}_MM_Lesion_seg_*", TRAINER_DIR)
        )
        if not trainer_dirs:
            print(f"Missing model: Dataset{dataset_id}")
            continue
        trainer_dir = trainer_dirs[0]
        model_name = os.path.basename(os.path.dirname(trainer_dir))

        plans = read_json(os.path.join(trainer_dir, "plans.json"))
        dataset = read_json(os.path.join(trainer_dir, "dataset.json"))
        config = plans["configurations"][CONFIGURATION]
        architecture = config["architecture"]
        arch_kwargs = architecture["arch_kwargs"]

        row = {
            "dataset_id": dataset_id,
            "model": model_name,
            "num_input_channels": len(dataset["channel_names"]),
            "input_channels": ", ".join(dataset["channel_names"].values()),
            "num_training_cases": dataset.get("numTraining"),
            "parameters": parameters.get(model_name),
            "patch_size": config["patch_size"],
            "batch_size": config["batch_size"],
            "spacing": config["spacing"],
            "median_image_size_in_voxels": config["median_image_size_in_voxels"],
            "normalization": sorted(set(config["normalization_schemes"])),
            "batch_dice": config.get("batch_dice"),
            "network": short_name(architecture["network_class_name"]),
            "n_stages": arch_kwargs["n_stages"],
            "features_per_stage": arch_kwargs["features_per_stage"],
            "kernel_sizes": arch_kwargs["kernel_sizes"],
            "strides": arch_kwargs["strides"],
            "n_conv_per_stage_encoder": arch_kwargs["n_conv_per_stage"],
            "n_conv_per_stage_decoder": arch_kwargs["n_conv_per_stage_decoder"],
            "conv_op": short_name(arch_kwargs["conv_op"]),
            "norm_op": short_name(arch_kwargs["norm_op"]),
            "nonlinearity": short_name(arch_kwargs["nonlin"]),
            "dropout": arch_kwargs["dropout_op"],
            "conv_bias": arch_kwargs["conv_bias"],
        }

        training_settings = read_training_settings(trainer_dir)
        if training_settings:
            for key in training_settings[0]:
                values = {str(setting[key]) for setting in training_settings}
                row[key] = training_settings[0][key] if len(values) == 1 else "varies across folds"
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No models found in {TRAINING_RESULTS_ROOT}")
    return pd.DataFrame(rows)


def build_summary(df):
    summary_rows = []
    for column in df.columns:
        if column in ("dataset_id", "model", "input_channels", "num_input_channels",
                      "num_training_cases", "parameters"):
            continue
        values = df[column].apply(lambda value: str(format_value(value)))
        counts = values.value_counts()
        summary_rows.append({
            "setting": column,
            "identical_across_models": len(counts) == 1,
            "value": counts.index[0] if len(counts) == 1 else "; ".join(
                f"{value} (n={count})" for value, count in counts.items()
            ),
        })
    return pd.DataFrame(summary_rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = build_configuration_rows()
    summary = build_summary(df)
    different = [row["setting"] for _, row in summary.iterrows() if not row["identical_across_models"]]

    for filename, table in (
        ("model_configurations.csv", df),
        ("model_configuration_summary.csv", summary),
    ):
        path = os.path.join(OUTPUT_DIR, filename)
        table.to_csv(path, index=False)
        print(f"Saved: {path}")

    print(f"Models processed: {len(df)}")
    print(f"Settings that differ across models: {different or 'none'}")


if __name__ == "__main__":
    main()
