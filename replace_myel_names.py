#!/usr/bin/env python3
"""
Script for renaming identifiers in JSON files
inside all subfolders of selected directories.

Base folder:
    /home/nohel/DATA/MultipleMyeloma_analyses/full_models

Target folders:
    longi_summary_all
    longi_summary_larger_than_0_3_cubic_cm
    longi_summary_larger_than_0_5_cubic_cm

The script scans all nested subfolders and processes all JSON files.
Uses temporary placeholders to avoid chained replacements.
"""

import argparse
import os
import sys

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
ANALYSIS_DIRS = [
    "longi_summary_all",
    "longi_summary_larger_than_0_3_cubic_cm",
    "longi_summary_larger_than_0_5_cubic_cm",
]
TARGETS = {
    "full_models": [os.path.join(ROOT, "full_models", folder) for folder in ANALYSIS_DIRS],
    "zero_input_models": [
        os.path.join(ROOT, "zero_input_models", folder) for folder in ANALYSIS_DIRS
    ],
    "iou_threshold_analysis": [
        os.path.join(ROOT, "iou_threshold_analysis_full_models", "longi_summary_all")
    ],
}

# Replacement mapping
replacements = {
    "Myel_069": "Myel_012_b",
    "Myel_012": "Myel_012_a",
    "Myel_047": "Myel_018_b",
    "Myel_018": "Myel_018_a",
    "Myel_024": "Myel_023_a",
    "Myel_023": "Myel_023_b",
    "Myel_059": "Myel_043_a",
    "Myel_043": "Myel_043_b",
    "Myel_070": "Myel_052_b",
    "Myel_052": "Myel_052_a",
}


def replace_content(content):
    """Apply replacements without chaining or renaming already replaced IDs."""
    placeholders = {}

    # Protect final IDs first because some begin with another source ID.
    for i, new in enumerate(replacements.values()):
        token = f"__FINAL_{i}__"
        placeholders[token] = new
        content = content.replace(new, token)

    # Replace original IDs with temporary tokens to prevent chained changes.
    for i, old in enumerate(replacements):
        token = f"__SOURCE_{i}__"
        placeholders[token] = replacements[old]
        content = content.replace(old, token)

    # Restore both protected final IDs and newly replaced IDs.
    for token, new in placeholders.items():
        content = content.replace(token, new)

    return content


def process_file(filepath):
    """Process one JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    updated_content = replace_content(content)
    if updated_content == content:
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(updated_content)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rename patient IDs in model-result JSONs.")
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="iou_threshold_analysis",
        help="Dataset tree to process (defaults to the IoU-threshold results).",
    )
    args = parser.parse_args(argv)

    processed = 0
    errors = []

    for root_dir in TARGETS[args.target]:
        if not os.path.exists(root_dir):
            print(f"Skipping missing folder: {root_dir}")
            continue

        print(f"\nScanning: {root_dir}")

        for current_path, _, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(current_path, filename)

                    try:
                        if process_file(filepath):
                            processed += 1

                    except Exception as e:
                        print(f"Error: {filepath}")
                        print(e)
                        errors.append(filepath)

    print(f"\nDone. Updated {processed} JSON files in target '{args.target}'.")
    if errors:
        print(f"Failed to process {len(errors)} JSON files.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())