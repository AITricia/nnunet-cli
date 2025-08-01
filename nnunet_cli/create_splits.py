from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


def create_splits_for_training_and_validation_files(
    image_dir: Path, output_path: Path, n_folds: int = 1
):
    # Load filenames and extract patient IDs
    # Assumes files are like: img3d_bavcta001_seg01_baseline_0000.nii.gz

    file_list = sorted(set(image_dir.glob("*.nii*")))

    df = pd.DataFrame(
        {
            "filename": [
                f.stem.replace("_0000", "").replace(".nii", "") for f in file_list
            ]
        }
    )

    # Extract consistent patient ID, e.g., bavcta001 from seg01_CT_bavcta001.nii.gz
    df["patient_id"] = df["filename"].str.extract(r"(bavcta\d+)", expand=False)

    if n_folds >= 2:
        # === CREATE FOLD ASSIGNMENTS SAFELY ===
        gkf = GroupKFold(n_splits=n_folds)
        df["fold"] = -1

        for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df["patient_id"])):
            df.loc[val_idx, "fold"] = fold

        # === BUILD SPLITS_FINAL.PKL ===
        splits = []
        for fold_id in range(n_folds):
            val_patients = df[df["fold"] == fold_id]["patient_id"].unique()
            train_patients = df[df["fold"] != fold_id]["patient_id"].unique()

            val_cases = df[df["patient_id"].isin(val_patients)]["filename"].tolist()
            train_cases = df[df["patient_id"].isin(train_patients)]["filename"].tolist()
            splits.append({"train": train_cases, "val": val_cases})
        print(f"Saved {n_folds}-fold patient-level splits to: {output_path}")

    # This will generate a single training/validation split
    # It identifies unique patient IDs, then splits those into training and validation sets to ensure no data leakage.
    else:
        unique_patients = df["patient_id"].unique()
        train_patients, val_patients = train_test_split(
            unique_patients, test_size=0.2, random_state=42
        )
        train_cases = df[df["patient_id"].isin(train_patients)]["filename"].tolist()
        val_cases = df[df["patient_id"].isin(val_patients)]["filename"].tolist()

        splits = [{"train": train_cases, "val": val_cases}]

    with open(output_path, "w") as f:
        json.dump(splits, f)

    print(f"Saved patient-level splits to: {output_path}")


def main():
    """
    Main function to generate patient-level splits for nnUNet training.
    It creates a splits_final.json file with the train/val assignments.
    """
    parser = ArgumentParser(
        description="Generate patient-level splits for nnUNet training."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to the directory containing training images.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of folds for cross-validation (default: 5).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory where splits_final.json will be saved.",
    )

    args = parser.parse_args()
    image_dir = Path(args.image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"The specified image directory '{image_dir}' does not exist. Please provide a valid path."
        )
    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"The specified image directory '{image_dir}' is not a directory. Please provide a valid directory."
        )

    n_folds = args.n_folds

    output_path = Path(args.output_path) / "splits_final.json"

    if output_path.exists():
        print(
            f"⚠️ Warning: The output file '{output_path}' already exists. It will be overwritten."
        )
    create_splits_for_training_and_validation_files(
        image_dir, output_path, n_folds=n_folds
    )
    print(f"✅ Patient-level splits created successfully and saved to '{output_path}'.")


if __name__ == "__main__":
    main()
