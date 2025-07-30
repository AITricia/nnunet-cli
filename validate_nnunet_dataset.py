import argparse
import subprocess
from pathlib import Path

# This completes some validation on Mac, but is much faster on Tethy's -- including for documentation only.
def validate_nnunet_dataset(dataset_dir: Path, dataset_num: int):
    """
    Validates a nnU-Net dataset using nnUNetv2_preprocess.

    Args:
        dataset_dir (Path): Path to the base nnUNet dataset directory.
        dataset_num (str): Number of the dataset
    
    Returns:
        str: Output log from the validation process.
    """
    
    if not dataset_dir.exists():
        return f"Dataset path {dataset_dir} does not exist."

    try:
        result = subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", dataset_num, "--verify_dataset_integrity"],
            cwd=dataset_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validate a nnU-Net dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the nnUNet dataset directory.")
    parser.add_argument("--dataset-num", type=str, required=True, help="Dataset number (e.g., 1 for Dataset1).")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"The dataset directory '{dataset_dir}' does not exist. Please provide a valid path.")
    
    output = validate_nnunet_dataset(dataset_dir, args.dataset_num)
    print(output)