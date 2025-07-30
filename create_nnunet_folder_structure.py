import argparse
from collections import defaultdict
import shutil
from pathlib import Path
from typing import List

from nnunet_cli.extract_labels_from_segmentation import extract_labels_from_segmentation

# Extracts the patient ID from filenames with the format: img3d_bavctaXXX_segYY_baseline.nii.gz
# Example: 'img3d_bavcta008_seg19_baseline.nii.gz' → returns 'bavcta008'
def get_patient_id(filename: Path):
    return filename.name.split('_')[1]  


def greedy_group_split(all_files, get_group_id_fn, test_fraction=0.2):
    """
    Greedily assigns small groups to the test set until the target number of files is reached.
    
    Args:
        all_files (List[Path]): List of image files.
        get_group_id_fn (Callable): Function that extracts group ID (e.g. patient ID) from file.
        test_fraction (float): Desired fraction of total files to go in test set.

    Returns:
        train_files (List[Path]), test_files (List[Path])
    """
    # Step 1: Group files by group ID
    group_to_files = defaultdict(list)
    for f in all_files:
        group_id = get_group_id_fn(f)
        group_to_files[group_id].append(f)

    # Step 2: Sort groups by number of files (ascending)
    sorted_groups = sorted(group_to_files.items(), key=lambda x: len(x[1]))

    # Step 3: Assign to test until test_fraction is met
    total_files = len(all_files)
    target_test_size = int(total_files * test_fraction)

    test_files, train_files = [], []
    test_count = 0

    for group_id, files in sorted_groups:
        if test_count + len(files) <= target_test_size:
            test_files.extend(files)
            test_count += len(files)
        else:
            train_files.extend(files)

    return train_files, test_files


def strip_suffix(filename: Path):
    return filename.name.replace(".nii.gz", "").replace(".nii", "")

def create_nnunet_folder_structure(images_dir: Path, segmentations_dir: Path, dataset_dir: Path, labels: List[int]):
    all_files = sorted(set(images_dir.glob("*.nii")).union(images_dir.glob("*.nii.gz")))

    for subfolder in ["imagesTs", "imagesTr", "labelsTr"]:
        (dataset_dir/subfolder).mkdir(parents=True, exist_ok=True)

    train_files, test_files = greedy_group_split(all_files, get_patient_id, test_fraction=0.2)

    for file in train_files:

        shutil.copy(file, f"{dataset_dir}/imagesTr/{strip_suffix(file)}_0000.nii.gz")
      
        segmentation_file = segmentations_dir / file.name
        if segmentation_file.exists():
            extract_labels_from_segmentation(segmentation_file, labels, f"{dataset_dir}/labelsTr/{strip_suffix(file)}.nii.gz")
        else:
            raise FileNotFoundError(f"Missing segmentation file: {segmentation_file}")


    for file in test_files:
        shutil.copy(file, dataset_dir/"imagesTs"/f"{strip_suffix(file)}_0000.nii.gz")
    print(f"✅ Copied {len(train_files)} training files and {len(test_files)} test files to the nnUNet folder structure at {dataset_dir}.")
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create nnUNet folder structure for a dataset.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing 3D CT images.")
    parser.add_argument("--segmentations-dir", type=Path, required=True, help="Directory containing 3D segmentations.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Output directory for nnUNet dataset structure.")
    
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory '{args.images_dir}' does not exist.")
    if not args.segmentations_dir.exists():
        raise FileNotFoundError(f"Segmentations directory '{args.segmentations_dir}' does not exist.")
    if not args.dataset_dir.exists():
        args.dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created dataset directory: {args.dataset_dir}")
    
    create_nnunet_folder_structure(args.images_dir, args.segmentations_dir, args.dataset_dir)

