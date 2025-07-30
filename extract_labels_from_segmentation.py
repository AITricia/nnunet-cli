import argparse
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np


def extract_labels_from_segmentation(segmentation_path: Path, labels: List[int], output_path: Path):
    # Load the segmentation file
    segmentation_img = nib.load(str(segmentation_path))
    # comment out if this is handled by pre-processing upstream (smoothing, etc.)
    segmentation_data = segmentation_img.get_fdata()
    # # Create a new NIfTI image for each label
    label_mask = np.isin(segmentation_data, labels).astype(np.uint8)
    label_img = nib.Nifti1Image(label_mask, affine=segmentation_img.affine)

    nib.save(label_img, str(output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract specific labels from a segmentation file.")
    parser.add_argument("--segmentation-path", type=Path, required=True, help="Path to the segmentation file (NIfTI format).")
    parser.add_argument("--labels", type=int, nargs='+', required=True, help="List of labels to extract, space-separated.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output path for the extracted label file.")

    args = parser.parse_args()

    extract_labels_from_segmentation(args.segmentation_path, args.labels, args.output_path)
    print(f"Extracted labels {args.labels} and saved to {args.output_path}.")