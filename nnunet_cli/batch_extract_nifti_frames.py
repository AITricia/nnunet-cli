"""
Automates extraction of:
  - 3D segmentations from segmentation NIfTI files
  - 3D CT images from 4D CT NIfTI files

Outputs:
  - 3D segmentations are saved in the user-specified output directory (--seg-3d-output-dir).
  - 3D CT images are saved in the user-specified output directory (--ct-3d-output-dir).

Workflow:
  1. Checks if the output directories exist; creates them if needed.
  2. For each segmentation:
      - Checks if the corresponding output files already exist.
      - If not, performs extraction for the missing file(s).
  3. Supports a dry-run mode to preview operations without modifying files.
"""

import argparse
from pathlib import Path
import warnings

import nibabel as nib
import numpy as np
from nnunet_cli.extract_3d_from_4d_nifti import (
    extract_3d_from_4d_nifti,
    ExtractionError,
)


def extract_3d_segmentation(seg_3d_file_path: Path, seg_3d_output_path: Path):
    # Rename the segmentation file for nnUNet compatibility, place in segmentation output folder
    img = nib.load(seg_3d_file_path)

    # Check for redundant dimensions
    singleton_axes = [i for i, dim in enumerate(img.shape) if dim == 1]
    if singleton_axes:
        warnings.warn(
            f"[WARNING] Found singleton axis/axes at {singleton_axes} in {seg_3d_file_path.name} with shape {img.shape}"
        )

    data = np.squeeze(
        img.get_fdata()
    )  # Handles the edge case where there is a singleton axis
    if data.ndim != 3:
        raise RuntimeError(f"{seg_3d_file_path} is still {data.shape}")
    new_img = nib.Nifti1Image(data.astype(img.get_data_dtype()), img.affine, img.header)
    nib.save(new_img, seg_3d_output_path)


def parse_segmentation_filename(filename: str):
    """
    Parses a segmentation filename and extracts key metadata.
    Assumes filenames like: segXX_CT_bavctaXXX_baseline[_rest].nii.gz

    Returns:
        seg_timeframe (str): e.g., "seg01"
        patient_id   (str): e.g., "bavcta001"
        timepoint    (str): e.g., "baseline"
        frame_index  (str): e.g., "01"  (from seg_timeframe)
    Raises:
        ValueError if the filename is malformed or frame index is not numeric.
    """
    meta = Path(filename).stem.replace(".nii", "").split("_")

    if len(meta) < 4:
        raise ValueError(f"Malformed filename: {filename}")
    seg_timeframe = meta[0]  # e.g., seg01
    patient_id = meta[2]  # e.g., bavcta001
    timepoint = meta[3]  # e.g., baseline
    if seg_timeframe.startswith("seg"):
        frame_index = seg_timeframe[3:]
    else:
        raise ValueError(f"Unrecognized seg_timeframe format: {seg_timeframe}")
    return seg_timeframe, patient_id, timepoint, frame_index


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D segmentations and CT files from 4D segmentation files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, will not perform the extraction, just print the intended output paths.",
    )
    parser.add_argument(
        "--input-seg-3d-dir",
        type=str,
        required=True,
        help="Directory containing 3D segmentation NIfTIs",
    )
    parser.add_argument(
        "--input-ct-4d-dir", type=str, help="Path to folder containing 4D CT files"
    )
    parser.add_argument(
        "--seg-3d-output-dir",
        type=str,
        help="Path to output folder for 3D segmentation files",
    )
    parser.add_argument(
        "--ct-3d-output-dir", type=str, help="Path to output folder for 3D CT files"
    )
    parser.add_argument(
        "--extract-ct-3d",
        action="store_true",
        help="Extract 3D frame from the 4D CT file",
    )
    parser.add_argument(
        "--extract-seg-3d",
        action="store_true",
        help="Extract 3D frame from the 3D segmentation file (remove singleton if necessary)",
    )

    args = parser.parse_args()

    dry_run = args.dry_run
    extract_seg_3d = args.extract_seg_3d
    extract_ct_3d = args.extract_ct_3d

    seg_3d_output_dir = Path(args.seg_3d_output_dir) if args.seg_3d_output_dir else None
    ct_3d_output_dir = Path(args.ct_3d_output_dir) if args.ct_3d_output_dir else None
    input_seg_3d_dir = Path(args.input_seg_3d_dir)
    input_ct_4d_dir = Path(args.input_ct_4d_dir) if args.input_ct_4d_dir else None

    if not (extract_seg_3d or extract_ct_3d):
        raise SystemExit(
            "⚠️ No extraction option specified (--extract-seg-3d or --extract-ct-3d). Nothing will be done."
        )

    if not args.input_seg_3d_dir:
        raise SystemExit("Error -- input-seg-3d-dir is required for this operation")

    if not input_seg_3d_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_seg_3d_dir}' does not exist.")
    if not input_seg_3d_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory for segmentations, got: {input_seg_3d_dir}"
        )

    if extract_seg_3d:
        if seg_3d_output_dir is None:
            raise SystemExit(
                "Error -- seg-3d-output-dir is required for this operation"
            )
        else:
            seg_3d_output_dir.mkdir(parents=True, exist_ok=True)

    segmentation_files = sorted(input_seg_3d_dir.glob("*.nii*"))

    if not segmentation_files:
        raise FileNotFoundError(
            f"No 3D segmentation files found in '{input_seg_3d_dir}'."
        )
    print(
        f"Found {len(segmentation_files)} 3D segmentation files in '{input_seg_3d_dir}'."
    )

    if extract_ct_3d:

        if not input_ct_4d_dir or not ct_3d_output_dir:
            missing = []
            if not input_ct_4d_dir:
                missing.append("input-ct-4d-dir")
            if not ct_3d_output_dir:
                missing.append("ct-3d-output-dir")
            raise SystemExit(
                f"Error -- {', '.join(missing)} is required for this operation"
            )

        if not input_ct_4d_dir.exists():
            raise FileNotFoundError(
                f"Input directory '{input_ct_4d_dir}' does not exist."
            )
        if not input_ct_4d_dir.is_dir():
            raise NotADirectoryError(
                f"Expected a directory for segmentations, got: {input_ct_4d_dir}"
            )
        ct_3d_output_dir.mkdir(parents=True, exist_ok=True)

    for seg_file in segmentation_files:
        try:
            seg_timeframe, patient_id, timepoint, frame_index = (
                parse_segmentation_filename(seg_file)
            )

            output_filename = f"img3d_{patient_id}_{seg_timeframe}_{timepoint}.nii.gz"

            if extract_seg_3d:
                seg_3d_output_path = seg_3d_output_dir / output_filename

                if dry_run:
                    print(
                        f"Dry run: Would copy '{seg_file}' to '{seg_3d_output_dir / output_filename}'."
                    )
                    continue
                if seg_3d_output_path.exists():
                    print(f"✅ Skipped (already exists): {seg_3d_output_path}")
                else:
                    extract_3d_segmentation(
                        seg_file, seg_3d_output_dir / output_filename
                    )

                    print(
                        f"✅ Successfully copied '{seg_file}' to '{seg_3d_output_dir / output_filename}'."
                    )

            if extract_ct_3d:
                ct_3d_output_path = ct_3d_output_dir / output_filename
                ct_4d_input_path = (
                    input_ct_4d_dir / f"img4D_CT_{patient_id}_{timepoint}.nii.gz"
                )
                # Extract 3D timepoints from 4D images only if they have not already been extracted
                if not ct_3d_output_path.exists() or dry_run:
                    extract_3d_from_4d_nifti(
                        ct_4d_input_path,
                        ct_3d_output_path,
                        frame_index,
                        dry_run=dry_run,
                    )
                else:
                    print(f"✅ Skipped (already exists): {ct_3d_output_path}")

        except ExtractionError as e:
            print(f"🫠 Error extracting from '{seg_file}': {e}")

        except FileNotFoundError as e:
            print(f"❌ Error: {e}. Skipping CT extraction for '{seg_file.name}'.")

        except Exception as e:
            print(f"❌ Unexpected error processing '{seg_file}': {e}")


if __name__ == "__main__":
    main()
