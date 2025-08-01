from pathlib import Path
import nibabel as nib
import numpy as np


def get_all_labels_in_segmentations(seg_dir):
    all_labels = set()
    for seg_file in Path(seg_dir).glob("*.nii*"):
        seg = nib.load(seg_file).get_fdata()
        all_labels.update(np.unique(seg).astype(int))
    return all_labels


def validate_label_consistency(labels_passed, segmentations_dir):
    segmentation_labels = get_all_labels_in_segmentations(segmentations_dir)
    labels_not_found_in_segmentations = labels_passed - segmentation_labels
    if labels_not_found_in_segmentations:
        raise ValueError(
            f"The labels argument contains labels not found in the segmentations: {labels_not_found_in_segmentations}"
        )
