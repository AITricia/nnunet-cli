import argparse
import json
from pathlib import Path


def create_dataset_json(output_path: Path, dataset_name: str, labels_dict: dict = {"background": 0, "aorta": 1}):
    num_training_files = len(list((output_path / "imagesTr").glob("*.nii.gz")))
    num_test_files = len(list((output_path / "imagesTs").glob("*.nii.gz")))
    dataset_json = {
        "name": dataset_name,
        "channel_names": {"0": "CT"},
        "labels": labels_dict,
        "numTraining": num_training_files,
        "numTest": num_test_files,
        "file_ending": ".nii.gz",
    }
    dataset_json_path = output_path / "dataset.json"
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Created dataset.json at: {dataset_json_path}")
    print(f"Contains {num_training_files} training files and {num_test_files} test files.")
    return dataset_json_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dataset.json for nnUNet.")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for the nnUNet dataset.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--labels-dict", type=str, default='{"background": 0, "aorta": 1}', help="Label dictionary as a JSON string. Default: {'background': 0, 'aorta': 1}")
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"The output path '{output_path}' does not exist. Please provide a valid path.")
    
    try:
        labels_dict = json.loads(args.labels_dict)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid labels_dict JSON: {args.labels_dict}")

    create_dataset_json(output_path, args.dataset_name, args.labels_dict)

