import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from nnunet_cli.extract_3d_from_4d import extract_3d_frame, ExtractionError

# Automates extracting both:
# - 3d segmentations from segmentation file
# - 3d CT files from original 4d file
# These will go into folders named "3d_segmentations" and "3d_CT", respectively
# When this util is called, it checks if the output folders exist, and if not, creates them.
# Then, it checks if the extracted file already exists in both output folders. If not, it runs the extraction script.



def main():
    parser = argparse.ArgumentParser(description="Extract 3D segmentations and CT files from 4D segmentation files.")
    parser.add_argument("--segmentation-dir", type=str, help="Path to folder containing 4D segmentations", default="/Users/tricialobo/Documents/UPenn/Independent_Study/4D_CT_Images_Converted/4D_CT_Manual_Segmentation_Tricia")
    parser.add_argument("--output-dir", type=str, help="Path to output folder for 3D files", default="/Users/tricialobo/Documents/UPenn/Independent_Study/Extracted_3D_Images_From_4D_Images")
    parser.add_argument("--ct-dir", type=str, required=False, help="Path to folder containing 4D CT files", default="/Users/tricialobo/Documents/UPenn/Independent_Study/4D_CT_Images_Converted")
    parser.add_argument("--dry-run", action="store_true", help="If set, will not perform the extraction, just print the intended output paths.")
    parser.add_argument("--extract-4d", type=bool, default=True, help="Extract 3D frame from the 4D CT file")
    parser.add_argument("--extract-seg", type=bool, default=True, help="Extract 3D frame from the 4D segmentation file")
    args = parser.parse_args()
    segmentation_dir = Path(args.segmentation_dir)
    output_dir = Path(args.output_dir)
    ct_dir = Path(args.ct_dir)
    dry_run = args.dry_run
    extract_seg = args.extract_seg
    extract_4d = args.extract_4d

    if not segmentation_dir.exists():
        raise FileNotFoundError(f"Input directory '{segmentation_dir}' does not exist.")
    
    if not segmentation_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for segmentations, got: {segmentation_dir}")
    
    if not ct_dir.exists():
        raise FileNotFoundError(f"CT directory '{ct_dir}' does not exist.")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    segmentation_files = sorted(segmentation_dir.glob("*.nii.gz"))
    if not segmentation_files:
        raise FileNotFoundError(f"No 4D segmentation files found in '{segmentation_dir}'.")
    print(f"Found {len(segmentation_files)} 4D segmentation files in '{segmentation_dir}'.")


    ct_output_dir = output_dir / "3d_CT"

    ct_output_dir.mkdir(parents=True, exist_ok=True)

    for seg_file in segmentation_files:
        try:
            # Starts with format like seg01_CT_bavcta001_baseline_ascending_aorta
            seg_file_metadata = seg_file.stem.split("_")
            seg_timeframe = seg_file_metadata[0]

            segment = seg_file_metadata[0]  # e.g., seg01
            patient_id = seg_file_metadata[2]  # e.g., bavcta001
            timepoint = seg_file_metadata[3]  # e.g., baseline

            output_filename = f"img3d_{patient_id}_{segment}_{timepoint}.nii.gz"
            # Rename the segmentation file for nnUNet compatibility, place in segmentation output folder
            seg_3d_output_path = output_dir / "3d_segmentations" 
            # / output_filename
            seg_3d_output_path.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"Dry run: Would copy '{seg_file}' to '{seg_3d_output_path / output_filename}'.")
            elif extract_seg:

                img = nib.load(seg_file)
                data = np.squeeze(img.get_fdata())  # drop singleton axes
                if data.ndim != 3:
                    raise RuntimeError(f"{seg_file} is still {data.shape}")
                new_img = nib.Nifti1Image(data.astype(img.get_data_dtype()),
                                        img.affine, img.header)
                nib.save(new_img, seg_3d_output_path / output_filename)
                print(f"✅ Successfully copied '{seg_file}' to '{seg_3d_output_path / output_filename}'.")

            if extract_4d:
                ct_3d_output_path = ct_output_dir / output_filename
                ct_4d_input_path = ct_dir / f"img4D_CT_{patient_id}_{timepoint}.nii.gz"

                if not ct_3d_output_path.exists() or dry_run:
                    timepoint_index = seg_timeframe[3:]
                    extract_3d_frame(ct_4d_input_path, ct_3d_output_path, timepoint_index, dry_run=dry_run)
                else:
                    print(f"✅ Skipped (already exists): {ct_3d_output_path.name}")

        except ExtractionError as e:
            print(f"🫠 Error extracting from '{seg_file}': {e}")
        
        except FileNotFoundError as e:
            print(f"❌ Error: {e}. Skipping CT extraction for '{seg_file.name}'.")
        except Exception as e:
            print(f"❌ Unexpected error processing '{seg_file}': {e}")

    print(f"Extracted 3D CT files to '{output_dir/'3D_CT'}'.")

    
if __name__ == '__main__':
    main()