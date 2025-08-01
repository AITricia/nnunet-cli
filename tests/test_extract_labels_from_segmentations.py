import numpy as np
import nibabel as nib
from pathlib import Path

from nnunet_cli.extract_labels_from_segmentation import extract_labels_from_segmentation


def create_mock_segmentation(path: Path, shape=(10, 10, 10), label_values=(0, 1, 2, 3)):
    """
    Create a mock segmentation NIfTI file with given label values.
    Each label is assigned to a quarter of the volume.
    """
    data = np.zeros(shape, dtype=np.uint8)
    z_len = shape[2] // len(label_values)
    for i, label in enumerate(label_values):
        data[:, :, i * z_len : (i + 1) * z_len] = label
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def test_extract_single_label(tmpdir):
    temp_dir = Path(tmpdir)
    input_file = temp_dir / "input.nii.gz"
    output_file = temp_dir / "output.nii.gz"

    create_mock_segmentation(input_file, shape=(5, 5, 4), label_values=(0, 1, 2, 3))
    extract_labels_from_segmentation(input_file, labels=[2], output_path=output_file)

    result = nib.load(output_file).get_fdata()
    assert set(np.unique(result)) == {0, 1}

    # Check that the 2-labeled region is preserved
    expected_mask = nib.load(input_file).get_fdata() == 2
    assert np.array_equal(result.astype(bool), expected_mask)


def test_extract_multiple_labels_preserves_labels(tmpdir):
    temp_dir = Path(tmpdir)
    input_file = temp_dir / "input.nii.gz"
    output_file = temp_dir / "output.nii.gz"

    create_mock_segmentation(input_file, shape=(5, 5, 4), label_values=(0, 1, 2, 3))
    extract_labels_from_segmentation(input_file, labels=[1, 3], output_path=output_file)

    result = nib.load(output_file).get_fdata()
    assert set(np.unique(result)) == {0, 1, 2}

    input_data = nib.load(input_file).get_fdata()
    expected_mask = np.isin(input_data, [1, 3])
    assert np.array_equal(result.astype(bool), expected_mask)


def test_extract_nonexistent_label_outputs_all_zeros(tmpdir):
    temp_dir = Path(tmpdir)
    input_file = temp_dir / "input.nii.gz"
    output_file = temp_dir / "output.nii.gz"

    create_mock_segmentation(input_file, shape=(5, 5, 5), label_values=(0, 1, 2))
    extract_labels_from_segmentation(input_file, labels=[9], output_path=output_file)

    result = nib.load(output_file).get_fdata()
    assert np.all(result == 0)


def test_output_shape_and_affine_and_header_preserved(tmpdir):
    temp_dir = Path(tmpdir)
    input_file = temp_dir / "input.nii.gz"
    output_file = temp_dir / "output.nii.gz"

    create_mock_segmentation(input_file, shape=(7, 7, 7), label_values=(0, 1))
    original_img = nib.load(input_file)
    extract_labels_from_segmentation(input_file, labels=[1], output_path=output_file)
    result_img = nib.load(output_file)

    assert result_img.shape == original_img.shape
    assert np.allclose(result_img.affine, original_img.affine)
    assert result_img.header == original_img.header
