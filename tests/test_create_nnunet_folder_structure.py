from pathlib import Path
from unittest.mock import patch

from nnunet_cli.create_nnunet_folder_structure import (
    get_patient_id,
    greedy_group_split,
    strip_suffix,
    create_nnunet_folder_structure,
)


def test_get_patient_id():
    assert get_patient_id(Path("img3d_bavcta001_seg01_baseline.nii.gz")) == "bavcta001"


def test_strip_suffix():
    assert strip_suffix(Path("image_001.nii.gz")) == "image_001"
    assert strip_suffix(Path("image_002.nii")) == "image_002"


def test_greedy_group_split_balanced():
    # Simulate 10 patients, each with 1 file
    files = [Path(f"img_patient{i:03d}_seg.nii.gz") for i in range(10)]

    def mock_get_group_id(path):
        return path.name.split("_")[1]  # patientXXX

    train, test = greedy_group_split(files, mock_get_group_id, test_fraction=0.3)

    assert len(test) == 3
    assert len(train) == 7
    assert set(train).isdisjoint(set(test))


@patch("nnunet_cli.create_nnunet_folder_structure.extract_labels_from_segmentation")
@patch("nnunet_cli.create_nnunet_folder_structure.shutil.copy")
def test_create_nnunet_folder_structure(
    mock_copy,
    mock_extract,
    dummy_3d_seg_dir_nnunet_input,
    dummy_3d_ct_dir_nnunet_input,
    tmp_path,
):
    output_dir = tmp_path / "nnunet_dataset"
    labels = [1]

    create_nnunet_folder_structure(
        dummy_3d_ct_dir_nnunet_input, dummy_3d_seg_dir_nnunet_input, output_dir, labels
    )

    assert (output_dir / "imagesTr").exists()
    assert (output_dir / "imagesTs").exists()
    assert (output_dir / "labelsTr").exists()

    assert mock_copy.called
    assert mock_extract.called
