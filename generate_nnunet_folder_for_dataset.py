"""
This script generates the folder structure required for training a nnUNet model from a 3D dataset extracted from 4D CT images.

Arguments:
    --output-path: Path where the nnUNet-compatible dataset structure will be created.
    --dataset-name: Name of the dataset (must follow 'DatasetXXX_Name' format, e.g. 'Dataset001_AortaCT').
    --n-folds: Number of cross-validation folds to create (default: 5).
    --create-splits: If set, creates patient-level training/validation splits (default: True).
    --images-dir: Path to the 3D images directory.
    --segmentations-dir: Path to the 3D segmentations directory.
    --labels: Space-separated list of label values to include 
    --labels_dict: Dictionary mapping label names to their corresponding values (as a JSON string).

    python generate_nnunet_folder_for_dataset.py \
  --images-dir /Users/tricialobo/Documents/UPenn/Independent_Study/Additional_segmentations_Tricia_not_prev_segmented_extracted/3d_CT \
  --segmentations-dir /Users/tricialobo/Documents/UPenn/Independent_Study/Additional_segmentations_Tricia_not_prev_segmented_extracted/3d_segmentations \
  --labels 7 \
  --output-path /Users/tricialobo/Documents/UPenn/Independent_Study/nnUNet/nnUNet_raw \
  --dataset-name Dataset101_AscendingAorta
  --labels-dict '{"background": 0, "aorta": 1}' \

Main Steps:
    1. Validates input paths and dataset name format.
    2. Sets up the nnUNet folder structure (imagesTr, labelsTr, etc.).
    3. Extracts and filters segmentations based on specified labels.
    4. Optionally creates cross-validation splits at the patient level.
    5. Generates a dataset.json file containing dataset metadata.
    6. Validates the resulting dataset using nnUNet's validation tools.

Supports future extension to multi-modality datasets (e.g., MRI).
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from nnunet_cli.create_splits import create_splits_for_training_and_validation_files
from nnunet_cli.validate_nnunet_dataset import validate_nnunet_dataset
from nnunet_cli.create_nnunet_folder_structure import create_nnunet_folder_structure
from nnunet_cli.create_dataset_json import create_dataset_json


def main():
    parser = argparse.ArgumentParser(description="Generate nnUNet folder structure for a 3D dataset extracted from 4D CT images.")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for the generated nnUNet folder.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for cross-validation (default: 5).")
    parser.add_argument("--create-splits", action='store_true', help="Create patient-level splits for training and validation.")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to the directory containing 3D CT images.")
    parser.add_argument("--segmentations-dir", type=str, required=True, help="Path to the directory containing 3D segmentations.")
    parser.add_argument("--labels", type=int, nargs='+', required=True, help="Labels to extract from the segmentations, space-separated.")
    parser.add_argument("--labels-dict", type=str, required=True,help='The labels dictionary to use in the dataset.json file. Should be a JSON string with label names and their corresponding values. For example: {"background": 0, "aorta": 1}.')
    parser.add_argument("--force", action="store_true", help="Overwrite existing output folder if it exists.")

    args = parser.parse_args()

    dataset_name = args.dataset_name

    match = re.match(r'Dataset(\d{3})_(\w+)', dataset_name)
    if not match:
        raise ValueError("Dataset name must follow the format 'DatasetXXX_Name', e.g. 'Dataset001_AortaCT'")
    dataset_num = match.group(1)
    if not dataset_num:
        raise ValueError(f"Dataset name '{dataset_name}' does not contain a valid dataset number. Please provide a name in the format 'DatasetXXX'.")
   
    # Create the Dataset output path if it doesn't exist
    dataset_output_path = Path(args.output_path)/dataset_name
    if dataset_output_path.exists():
        if args.force:
            print(f"⚠️ Output path '{dataset_output_path}' exists. Deleting it because --force was specified...")
            shutil.rmtree(dataset_output_path)
        else:
            print(f"❌ Output path '{dataset_output_path}' already exists. Use --force to overwrite.")
            sys.exit(1)

    # Verify that 3D image and 3D segmentation directories exist
    images_dir = Path(args.images_dir)
    segmentations_dir = Path(args.segmentations_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"The 3D images directory '{images_dir}' does not exist. Please provide a valid path.")
    if not segmentations_dir.exists():
        raise FileNotFoundError(f"The 3D segmentations directory '{segmentations_dir}' does not exist. Please provide a valid path.")
    
    # Create nnUNet folder structure
    # This will create the imagesTr, imagesTs, labelsTr, and labelsTs folders, splitting files between training and test sets.
    print(f"Creating nnUNet folder structure at {dataset_output_path}...")

    # Use provided labels to extract from the 3D segmentation files
    labels = args.labels

    create_nnunet_folder_structure(images_dir, segmentations_dir, dataset_output_path, labels)

    # Create training splits for cross-validation
    if args.create_splits:
        print("Creating patient-level splits...")
        create_splits_for_training_and_validation_files(dataset_output_path/"imagesTr", dataset_output_path / "splits_final.json", n_folds=args.n_folds)
        print("✅ Patient-level splits created successfully.")

    # Load labels dict and create dataset.json
    try:
        labels_dict = json.loads(args.labels_dict)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid labels_dict JSON: {args.labels_dict}")

    
    print("Creating dataset.json...")
    create_dataset_json(dataset_output_path, args.dataset_name, labels_dict)

    # Step 5: Validate the dataset (Can comment out as it's faster to complete validation on Tethy's - including for documentation)
    print(f"Validating nnUNet dataset {dataset_name}...")
    validation_result = validate_nnunet_dataset(dataset_output_path, dataset_num)
    print(f"Validation result: {validation_result}")

if __name__ == "__main__":
    main()