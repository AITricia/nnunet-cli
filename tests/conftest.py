import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def dummy_3d_seg_dir(tmp_path):
    seg_input = tmp_path / "seg"
    seg_input.mkdir()
    data = np.random.rand(4, 4, 4)
    nii = nib.Nifti1Image(data, np.eye(4))
    path = seg_input / "seg01_CT_bavcta001_baseline.nii.gz"
    nib.save(nii, path)
    return path.parent


@pytest.fixture
def dummy_4d_ct_dir(tmp_path):
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    data = np.random.rand(4, 4, 4, 20)
    nii = nib.Nifti1Image(data, np.eye(4))
    path = ct_dir / "img4D_CT_bavcta001_baseline.nii.gz"
    nib.save(nii, path)
    return path.parent


@pytest.fixture
def dummy_3d_ct_dir(tmp_path):
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    data = np.random.rand(4, 4, 4, 20)
    nii = nib.Nifti1Image(data, np.eye(4))
    path = ct_dir / "img3d_bavcta001_seg01_baseline.nii.gz"
    nib.save(nii, path)
    return path.parent


@pytest.fixture
def dummy_3d_ct_dir_nnunet_input(tmp_path):
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()

    # Create a fake 4D volume with 20 time points
    data = np.random.rand(4, 4, 4, 20)
    affine = np.eye(4)

    for i in range(20):
        # Extract individual 3D frame
        data_3d = data[:, :, :, i]
        nii = nib.Nifti1Image(data_3d, affine)

        # Filename must be compatible with your naming logic
        filename = f"img3d_bavcta{i:03d}_seg01_baseline.nii.gz"
        path = ct_dir / filename
        nib.save(nii, path)

    return ct_dir


@pytest.fixture
def dummy_3d_seg_dir_nnunet_input(tmp_path):
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()

    # Create a fake 4D volume with 20 time points

    for i in range(20):
        data = np.random.rand(4, 4, 4)
        data[0, 0, 0] = 1  # inject label 1
        affine = np.eye(4)
        # Extract individual 3D frame
        nii = nib.Nifti1Image(data, affine)

        # Filename must be compatible with your naming logic
        filename = f"img3d_bavcta{i:03d}_seg01_baseline.nii.gz"
        path = seg_dir / filename
        nib.save(nii, path)

    return seg_dir


@pytest.fixture
def dummy_nnunet_output_dir(tmp_path):
    imagesTr = tmp_path / "imagesTr"
    imagesTs = tmp_path / "imagesTs"
    imagesTr.mkdir()
    imagesTs.mkdir()

    # Create dummy training files
    for i in range(4):
        (imagesTr / f"img_{i:03d}_0000.nii.gz").touch()

    # Create dummy test files
    for i in range(2):
        (imagesTs / f"img_test_{i:03d}_0000.nii.gz").touch()

    return tmp_path
