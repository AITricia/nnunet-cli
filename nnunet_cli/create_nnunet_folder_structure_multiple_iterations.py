# First, divide the dataset into the various iterations:
# Start with the extracted 3d frames: /Users/tricialobo/Documents/UPenn/Independent_Study/Extracted_Frames_For_nnUNet
# Obtain a list of patient IDs, then use GroupKFold to separate
# Use shutil to move the files (3dseg, 4dct) into their own groupings
# Then, call create_nnunet_folder_structure.py to create the datasets
import argparse
from collections import defaultdict
import tempfile

from pathlib import Path


def get_patient_id(filename: Path):
    return filename.name.split("_")[1]


def greedy_group_split(all_files, get_group_id_fn, test_fraction=0.33):
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


def main(images_dir, segmentations_dir, num_iterations, dataset_names):

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create nnUNet folder structure for multiple iterations (datasets)."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to the directory containing 3D CT images.",
    )
    parser.add_argument(
        "--segmentations-dir",
        type=str,
        required=True,
        help="Path to the directory containing 3D segmentations.",
    )
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--dataset-names", type=str, nargs="+")

    args = parser.parse_args()

    images_dir = args.images_dir
    segmentations_dir = args.segmentations_dir
    num_iterations = args.num_iterations
    dataset_names = args.dataset_names

    main(images_dir, segmentations_dir, num_iterations, dataset_names)
