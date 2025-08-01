from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from nnunet_cli.batch_extract_nifti_frames import (
    extract_3d_segmentation,
    parse_segmentation_filename,
)

# --- UNIT TESTS ---


def test_parse_segmentation_filename_good():
    seg_timeframe, patient_id, timepoint, frame_index = parse_segmentation_filename(
        "seg01_CT_bavcta001_baseline.nii.gz"
    )
    assert seg_timeframe == "seg01"
    assert patient_id == "bavcta001"
    assert timepoint == "baseline"
    assert frame_index == "01"


def test_parse_segmentation_filename_bad():
    with pytest.raises(ValueError):
        parse_segmentation_filename("badfilename.nii.gz")


def test_extract_3d_segmentation(tmp_path):
    # Create fake 3D NIfTI image
    arr = np.random.rand(8, 8, 8)
    affine = np.eye(4)
    nii = nib.Nifti1Image(arr, affine)
    input_path = tmp_path / "seg_test.nii.gz"
    output_path = tmp_path / "out.nii.gz"
    nib.save(nii, input_path)

    extract_3d_segmentation(input_path, output_path)
    loaded = nib.load(output_path)
    assert loaded.shape == (8, 8, 8)
    np.testing.assert_allclose(loaded.get_fdata(), arr, atol=1e-6)


def test_warn_if_3d_seg_has_redundant_dimensions(tmp_path):
    with pytest.warns(UserWarning):
        # Create fake 3D NIfTI image
        arr = np.random.rand(1, 8, 8, 8)
        arr_squeezed = np.squeeze(arr)
        affine = np.eye(4)
        nii = nib.Nifti1Image(arr, affine)
        input_path = tmp_path / "seg_test.nii.gz"
        output_path = tmp_path / "out.nii.gz"
        nib.save(nii, input_path)

        extract_3d_segmentation(input_path, output_path)
        loaded = nib.load(output_path)
        assert loaded.shape == (8, 8, 8)
        np.testing.assert_allclose(loaded.get_fdata(), arr_squeezed, atol=1e-6)


# --- INTEGRATION TESTS --- #


def test_cli_dry_run(tmp_path, dummy_3d_seg_dir, dummy_4d_ct_dir):
    # Prepare a fake input directory
    seg_input = dummy_3d_seg_dir

    seg_3d_output = tmp_path / "seg_3d_output"
    seg_3d_output.mkdir()

    ct_3d_output = tmp_path / "ct_output"
    ct_3d_output.mkdir()

    script_path = (
        Path(__file__).parent.parent / "nnunet_cli/batch_extract_nifti_frames.py"
    )

    # Call CLI in dry-run mode
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--input-seg-3d-dir",
            str(seg_input),
            "--input-ct-4d-dir",
            str(dummy_4d_ct_dir),
            "--seg-3d-output-dir",
            str(seg_3d_output),
            "--ct-3d-output-dir",
            str(ct_3d_output),
            "--extract-seg-3d",
            "--extract-ct-3d",
            "--dry-run",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    # Should NOT create output files in dry run
    seg_3d_output_files = list(seg_3d_output.glob("*.nii.gz"))
    assert len(seg_3d_output_files) == 0

    ct_3d_output_files = list(ct_3d_output.glob("*.nii.gz"))
    assert len(ct_3d_output_files) == 0
    print(result.stdout)
    assert "Dry run" in result.stdout


def test_happy_path_extract_ct_and_seg_files(
    tmp_path, dummy_3d_seg_dir, dummy_4d_ct_dir
):
    # Prepare a fake input directory
    seg_input = dummy_3d_seg_dir

    seg_3d_output = tmp_path / "seg_3d_output"
    seg_3d_output.mkdir()

    ct_3d_output = tmp_path / "ct_output"
    ct_3d_output.mkdir()

    script_path = (
        Path(__file__).parent.parent / "nnunet_cli/batch_extract_nifti_frames.py"
    )

    # Call CLI in dry-run mode
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "--input-seg-3d-dir",
            str(seg_input),
            "--input-ct-4d-dir",
            str(dummy_4d_ct_dir),
            "--seg-3d-output-dir",
            str(seg_3d_output),
            "--ct-3d-output-dir",
            str(ct_3d_output),
            "--extract-seg-3d",
            "--extract-ct-3d",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )

    assert result.returncode == 0
    seg_output_files = list(seg_3d_output.glob("*.nii.gz"))
    assert len(seg_output_files) == 1

    ct_output_files = list(ct_3d_output.glob("*.nii.gz"))
    assert len(ct_output_files) == 1


@pytest.mark.parametrize(
    "extract_seg_3d, extract_ct_3d, expect_success, pass_seg_3d_dir, pass_ct_4d_dir, pass_seg_3d_output_dir, pass_ct_3d_output_dir, error_msg",
    [
        (None, None, False, True, True, True, False, "No extraction option specified"),
        (
            None,
            "--extract-ct-3d",
            False,
            False,
            True,
            True,
            False,
            "the following arguments are required: --input-seg-3d-dir",
        ),
        ("--extract-seg-3d", "", True, True, True, True, True, None),
        ("--extract-seg-3d", "", True, True, False, True, False, None),
        (
            None,
            "--extract-ct-3d",
            False,
            True,
            False,
            True,
            False,
            "input-ct-4d-dir, ct-3d-output-dir is required for this operation",
        ),
    ],
)
def test_cli_missing_argument_dependencies(
    tmp_path,
    dummy_3d_seg_dir,
    extract_seg_3d,
    extract_ct_3d,
    expect_success,
    pass_seg_3d_dir,
    pass_ct_4d_dir,
    pass_seg_3d_output_dir,
    pass_ct_3d_output_dir,
    error_msg,
):
    script_path = (
        Path(__file__).parent.parent / "nnunet_cli/batch_extract_nifti_frames.py"
    )

    cmd = [sys.executable, script_path]

    if extract_seg_3d:
        cmd.append(extract_seg_3d)
    if extract_ct_3d:
        cmd.append(extract_ct_3d)

    ct_4d_input = tmp_path / "ct"
    ct_4d_input.mkdir()

    ct_3d_output = tmp_path / "ct_output"
    ct_3d_output.mkdir()

    seg_3d_output = tmp_path / "seg_3d_output"
    seg_3d_output.mkdir()

    if pass_seg_3d_dir:
        cmd += ["--input-seg-3d-dir", str(dummy_3d_seg_dir)]

    if pass_ct_4d_dir:
        cmd += ["--input-ct-4d-dir", str(ct_4d_input)]

    if pass_ct_3d_output_dir:
        cmd += ["--ct-3d-output-dir", str(ct_3d_output)]

    if pass_seg_3d_output_dir:
        cmd += ["--seg-3d-output-dir", str(seg_3d_output)]

    if expect_success:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        assert result.returncode == 0
    else:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        assert result.returncode != 0
        assert error_msg in result.stderr
