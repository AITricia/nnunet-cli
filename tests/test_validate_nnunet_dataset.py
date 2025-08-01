import subprocess

from pathlib import Path
from unittest.mock import patch, MagicMock
from nnunet_cli.validate_nnunet_dataset import validate_nnunet_dataset


def test_validate_nnunet_dataset_success(tmp_path):
    dataset_dir = tmp_path / "Dataset001_Example"
    dataset_dir.mkdir(parents=True)

    mock_output = "Dataset integrity verified successfully."

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = mock_output
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        output = validate_nnunet_dataset(dataset_dir, 1)

        mock_run.assert_called_once_with(
            ["nnUNetv2_plan_and_preprocess", "-d", 1, "--verify_dataset_integrity"],
            cwd=dataset_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        assert output == mock_output


def test_validate_nnunet_dataset_path_does_not_exist():
    nonexistent_dir = Path("/some/fake/path")
    result = validate_nnunet_dataset(nonexistent_dir, 1)
    assert str(nonexistent_dir) in result
    assert "does not exist" in result


def test_validate_nnunet_dataset_process_error(tmp_path):
    dataset_dir = tmp_path / "Dataset001_Fail"
    dataset_dir.mkdir(parents=True)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["nnUNetv2_plan_and_preprocess"],
            stderr="Mocked error: dataset integrity failed.",
        )

        result = validate_nnunet_dataset(dataset_dir, 1)
        assert "dataset integrity failed" in result
