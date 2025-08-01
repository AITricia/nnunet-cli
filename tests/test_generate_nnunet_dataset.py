import json
import subprocess

from pathlib import Path
from unittest.mock import patch


@patch("nnunet_cli.generate_nnunet_dataset.validate_nnunet_dataset")
def test_happy_path_generate_dataset_single_fold(
    mock_validate, tmp_path, dummy_3d_ct_dir_nnunet_input, dummy_3d_seg_dir_nnunet_input
):
    script_path = Path(__file__).parent.parent / "nnunet_cli/generate_nnunet_dataset.py"
    output_path = tmp_path / "nnUNet_raw"
    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),
            "--segmentations-dir",
            str(dummy_3d_seg_dir_nnunet_input),
            "--labels",
            "1",
            "--output-path",
            str(output_path),
            "--dataset-name",
            "Dataset101_AscendingAorta",
            "--labels-dict",
            '{"background": 0, "aorta": 1}',
        ],
        capture_output=True,
        text=True,
    )

    # Check return code
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check output folders
    dataset_folder = output_path / "Dataset101_AscendingAorta"
    assert (dataset_folder / "imagesTr").exists()
    assert (dataset_folder / "labelsTr").exists()
    assert (dataset_folder / "dataset.json").exists()
    training_files = sorted((dataset_folder / "imagesTr").glob("*.nii.gz"))
    assert len(training_files) == 16
    test_files = sorted((dataset_folder / "imagesTs").glob("*.nii.gz"))
    assert len(test_files) == 4

    # Confirm no overlap between training and testing files
    overlapping_train_and_test_files = set(training_files).intersection(set(test_files))
    assert (
        len(overlapping_train_and_test_files) == 0
    ), f"Train and test overlap detected: {overlapping_train_and_test_files}"

    # Confirm that splits.json was created
    with open(dataset_folder / "splits_final.json") as f:
        splits_final = json.load(f)

    train_files = splits_final[0]["train"]
    assert len(train_files) == 12

    validation_files = splits_final[0]["val"]
    assert len(validation_files) == 4

    overlapping_train_and_validation_files = set(train_files).intersection(
        set(validation_files)
    )
    assert (
        len(overlapping_train_and_validation_files) == 0
    ), f"Train and validation overlap detected: {overlapping_train_and_validation_files}"

    # Check dataset.json content
    with open(dataset_folder / "dataset.json") as f:
        dataset_json = json.load(f)
    assert "labels" in dataset_json
    assert "aorta" in dataset_json["labels"]
    assert "background" in dataset_json["labels"]
    assert dataset_json["numTraining"] == 16
    assert dataset_json["numTest"] == 4
    assert dataset_json["name"] == "Dataset101_AscendingAorta"


def test_happy_path_generate_dataset_multi_fold(
    tmp_path, dummy_3d_ct_dir_nnunet_input, dummy_3d_seg_dir_nnunet_input
):
    script_path = Path(__file__).parent.parent / "nnunet_cli/generate_nnunet_dataset.py"
    output_path = tmp_path / "nnUNet_raw"

    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),
            "--segmentations-dir",
            str(dummy_3d_seg_dir_nnunet_input),
            "--labels",
            "1",
            "--output-path",
            str(output_path),
            "--dataset-name",
            "Dataset101_AscendingAorta",
            "--n-folds",
            "5",
            "--labels-dict",
            '{"background": 0, "aorta": 1}',
        ],
        capture_output=True,
        text=True,
    )

    # Check return code
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check output folders
    dataset_folder = output_path / "Dataset101_AscendingAorta"
    assert (dataset_folder / "imagesTr").exists()
    assert (dataset_folder / "labelsTr").exists()
    assert (dataset_folder / "dataset.json").exists()
    training_files = sorted((dataset_folder / "imagesTr").glob("*.nii.gz"))
    assert len(training_files) == 16
    test_files = sorted((dataset_folder / "imagesTs").glob("*.nii.gz"))
    assert len(test_files) == 4

    # Confirm no overlap between training and testing files
    overlapping_train_and_test_files = set(training_files).intersection(set(test_files))
    assert (
        len(overlapping_train_and_test_files) == 0
    ), f"Train and test overlap detected: {overlapping_train_and_test_files}"

    # Confirm that splits.json was created
    with open(dataset_folder / "splits_final.json") as f:
        splits_final = json.load(f)

    assert len(splits_final) == 5

    for fold in splits_final:
        train_files = fold["train"]
        validation_files = fold["val"]
        overlapping_train_and_validation_files = set(train_files).intersection(
            set(validation_files)
        )
        assert (
            len(overlapping_train_and_validation_files) == 0
        ), f"Train and validation overlap detected: {overlapping_train_and_validation_files}"

    # Check dataset.json content
    with open(dataset_folder / "dataset.json") as f:
        dataset_json = json.load(f)
    assert "labels" in dataset_json
    assert "aorta" in dataset_json["labels"]
    assert "background" in dataset_json["labels"]
    assert dataset_json["numTraining"] == 16
    assert dataset_json["numTest"] == 4
    assert dataset_json["name"] == "Dataset101_AscendingAorta"


def test_missing_required_arguments(tmp_path):
    script_path = Path(__file__).parent.parent / "nnunet_cli/generate_nnunet_dataset.py"
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "usage:" in result.stderr.lower()


def test_invalid_labels_dict(
    tmp_path, dummy_3d_ct_dir_nnunet_input, dummy_3d_seg_dir_nnunet_input
):
    script_path = Path(__file__).parent.parent / "nnunet_cli/generate_nnunet_dataset.py"
    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),
            "--segmentations-dir",
            str(dummy_3d_seg_dir_nnunet_input),
            "--labels",
            "1",
            "--output-path",
            str(tmp_path / "out"),
            "--dataset-name",
            "Dataset101_BadJSON",
            "--labels-dict",
            '{"background": 0, "aorta":',  # Malformed JSON
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Expecting value" in result.stderr or "json" in result.stderr.lower()


def test_missing_segmentation_file(tmp_path, dummy_3d_ct_dir_nnunet_input):
    # Create empty seg folder (no matching files)
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()

    script_path = Path(__file__).parent.parent / "nnunet_cli/generate_nnunet_dataset.py"
    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),
            "--segmentations-dir",
            str(seg_dir),
            "--labels",
            "1",
            "--output-path",
            str(tmp_path / "out"),
            "--dataset-name",
            "Dataset101_MissingSegs",
            "--labels-dict",
            '{"background": 0, "aorta": 1}',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "Missing segmentation file" in result.stderr
        or "not found" in result.stderr.lower()
    )


def test_invalid_n_folds_argument(
    tmp_path, dummy_3d_ct_dir_nnunet_input, dummy_3d_seg_dir_nnunet_input
):
    script_path = Path(__file__).parent.parent / "nnunet_cli/generate_nnunet_dataset.py"
    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),
            "--segmentations-dir",
            str(dummy_3d_seg_dir_nnunet_input),
            "--labels",
            "1",
            "--output-path",
            str(tmp_path / "out"),
            "--dataset-name",
            "Dataset101_BadFold",
            "--labels-dict",
            '{"background": 0, "aorta": 1}',
            "--n-folds",
            "-5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "n-folds" in result.stderr.lower() or "must be >=" in result.stderr.lower()


def test_force_overwrite(
    tmp_path, dummy_3d_ct_dir_nnunet_input, dummy_3d_seg_dir_nnunet_input
):
    script_path = Path(__file__).parent.parent / "nnunet_cli/generate_nnunet_dataset.py"
    output_path = tmp_path / "nnUNet_raw"

    # First run: creates dataset
    subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),
            "--segmentations-dir",
            str(dummy_3d_seg_dir_nnunet_input),
            "--labels",
            "1",
            "--output-path",
            str(output_path),
            "--dataset-name",
            "Dataset101_ForceTest",
            "--labels-dict",
            '{"background": 0, "aorta": 1}',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Add dummy file that should be removed on overwrite
    dummy_file = output_path / "Dataset101_ForceTest" / "this_should_be_deleted.txt"
    dummy_file.write_text("DELETE ME")

    assert dummy_file.exists(), "Setup failed: dummy file not created."

    # Second run: overwrite with --force
    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),
            "--segmentations-dir",
            str(dummy_3d_seg_dir_nnunet_input),
            "--labels",
            "1",
            "--output-path",
            str(output_path),
            "--dataset-name",
            "Dataset101_ForceTest",
            "--labels-dict",
            '{"background": 0, "aorta": 1}',
            "--force",
        ],
        capture_output=True,
        text=True,
    )
    assert not dummy_file.exists(), "Overwrite failed: dummy file still exists."
    assert result.returncode == 0, f"Force overwrite failed: {result.stderr}"
    assert (output_path / "Dataset101_ForceTest" / "dataset.json").exists()


def test_verify_labels_fails_with_unexpected_label(
    dummy_3d_ct_dir_nnunet_input, dummy_3d_seg_dir_nnunet_input, tmp_path
):
    script_path = (
        Path(__file__).parent.parent / "nnunet_cli" / "generate_nnunet_dataset.py"
    )
    output_path = tmp_path / "nnUNet_raw"

    result = subprocess.run(
        [
            "python",
            str(script_path),
            "--images-dir",
            str(dummy_3d_ct_dir_nnunet_input),  # just reuse seg dir
            "--segmentations-dir",
            str(dummy_3d_seg_dir_nnunet_input),
            "--labels",
            "3",
            "6",
            "--labels-dict",
            '{"background": 0, "aorta": 1}',  # only label 1 declared
            "--output-path",
            str(output_path),
            "--dataset-name",
            "Dataset101_Test",
            "--verify-labels",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert (
        "The labels argument contains labels not found in the segmentations: {3, 6}"
        in result.stderr
    )
