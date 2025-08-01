import pytest
from pathlib import Path

from nnunet_cli.utils.label_utils import (
    get_all_labels_in_segmentations,
    validate_label_consistency,
)


def test_get_all_labels_in_segmentations(dummy_3d_seg_dir_nnunet_input):
    labels = get_all_labels_in_segmentations(dummy_3d_seg_dir_nnunet_input)
    assert labels == {0, 1}


def test_validate_label_consistency_pass(dummy_3d_seg_dir_nnunet_input):
    # All labels are present in segmentation
    validate_label_consistency({1}, dummy_3d_seg_dir_nnunet_input)


def test_validate_label_consistency_raises(dummy_3d_seg_dir_nnunet_input):
    with pytest.raises(ValueError) as e:
        validate_label_consistency({3, 4}, dummy_3d_seg_dir_nnunet_input)

    assert "The labels argument contains labels not found in the segmentations" in str(
        e.value
    )
    assert "3" in str(e.value) and "4" in str(e.value)
