import nibabel as nib
import numpy as np
import pytest

from nnunet_cli.extract_3d_from_4d_nifti import (
    extract_3d_from_4d_nifti,
    ExtractionError,
)


def test_extract_ct_3d_frame_happy_path(tmp_path):
    data = np.random.rand(5, 5, 5, 10)  # 4D
    nii = nib.Nifti1Image(data, np.eye(4))
    input_file = tmp_path / "img4d_CT_bavcta010_baseline.nii.gz"
    output_file = tmp_path / "output.nii.gz"
    nib.save(nii, input_file)

    extract_3d_from_4d_nifti(input_file, output_file, time_index=3)

    out = nib.load(output_file).get_fdata()
    np.testing.assert_array_almost_equal(out, data[:, :, :, 2])


def test_raises_for_3d_files(tmp_path):
    data = np.random.rand(5, 5, 5)  # 4D
    nii = nib.Nifti1Image(data, np.eye(4))
    input_file = tmp_path / "img3d_CT_bavcta010_baseline.nii.gz"
    output_file = tmp_path / "output.nii.gz"
    nib.save(nii, input_file)
    with pytest.raises(ExtractionError, match="not a 4D NIfTI file"):
        extract_3d_from_4d_nifti(input_file, output_file, time_index=3)


def test_raises_on_out_of_range_index(tmp_path):
    data = np.random.rand(10, 10, 10, 5)
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    input_path = tmp_path / "img_CT_bavcta001_baseline.nii.gz"
    output_path = tmp_path / "output.nii.gz"
    nib.save(nii, input_path)

    with pytest.raises(ExtractionError, match="out of range"):
        extract_3d_from_4d_nifti(input_path, output_path, time_index=10)


def test_raises_on_bad_filename(tmp_path):
    data = np.random.rand(10, 10, 10, 5)
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    input_path = tmp_path / "badname.nii.gz"
    output_path = tmp_path / "output.nii.gz"
    nib.save(nii, input_path)

    with pytest.raises(ExtractionError, match="does not contain enough metadata"):
        extract_3d_from_4d_nifti(input_path, output_path, time_index=1)
