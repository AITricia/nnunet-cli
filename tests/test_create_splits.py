import json
from pathlib import Path

import numpy as np
import nibabel as nib

from nnunet_cli.create_splits import create_splits_for_training_and_validation_files


def create_mock_nifti(path: Path, patient_id: str, index: int):
    data = np.random.rand(4, 4, 4)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    filename = f"img3d_{patient_id}_seg01_baseline_{index:04d}.nii.gz"
    nib.save(img, path / filename)


def test_create_splits_for_training_and_validation_files_single_fold(tmpdir):
    tmp_path = Path(tmpdir)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_path = tmp_path / "splits.json"

    for i in range(10):
        create_mock_nifti(image_dir, f"bavcta{i:03d}", 0)

    create_splits_for_training_and_validation_files(image_dir, output_path, n_folds=1)

    with open(output_path) as f:
        splits = json.load(f)

    assert isinstance(splits, list)
    assert len(splits) == 1
    assert "train" in splits[0]
    assert "val" in splits[0]
    total_cases = len(splits[0]["train"]) + len(splits[0]["val"])
    assert total_cases == 10
    assert len(set(splits[0]["train"]).intersection(set(splits[0]["val"]))) == 0


def test_create_splits_for_training_and_validation_files_multiple_folds(tmpdir):
    tmp_path = Path(tmpdir)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_path = tmp_path / "splits.json"

    for i in range(10):
        create_mock_nifti(image_dir, f"bavcta{i:03d}", 0)

    create_splits_for_training_and_validation_files(image_dir, output_path, n_folds=5)

    with open(output_path) as f:
        splits = json.load(f)

    assert len(splits) == 5
    for split in splits:
        assert "train" in split
        assert "val" in split
        assert len(set(split["train"]).intersection(set(split["val"]))) == 0
