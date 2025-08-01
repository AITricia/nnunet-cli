import json
from nnunet_cli.create_dataset_json import create_dataset_json


def test_create_dataset_json_creates_correct_file(dummy_nnunet_output_dir):
    dataset_name = "TestDataset"
    labels_dict = {"background": 0, "aorta": 1}
    json_path = create_dataset_json(dummy_nnunet_output_dir, dataset_name, labels_dict)

    assert json_path.exists(), "dataset.json was not created"

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["name"] == dataset_name
    assert data["channel_names"] == {"0": "CT"}
    assert data["labels"] == labels_dict
    assert data["numTraining"] == 4
    assert data["numTest"] == 2
    assert data["file_ending"] == ".nii.gz"
