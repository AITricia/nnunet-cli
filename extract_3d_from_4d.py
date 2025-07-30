import argparse
from pathlib import Path

import nibabel as nib

class ExtractionError(Exception):
    pass

def extract_3d_frame(input_nifti: Path, output_nifti: Path, time_index: int = None, dry_run: bool=False):
    """
    Extracts a specific time frame (3D volume) from a 4D NIfTI file and saves it as a new file.
    
    Parameters:
        input (str): Path to the 4D NIfTI file.
        output (str): Output file name for the extracted 3D NIfTI file.
        
    """
    nii_4d = nib.load(input_nifti)
    data_4d = nii_4d.get_fdata()

    file_metadata = input_nifti.stem.split("_")
    
    if len(file_metadata) < 4:
        raise ExtractionError("Filename does not contain enough metadata for patient ID and timepoint.")

    output_path = output_nifti 


    # Ensure the image has a time dimension
    if len(data_4d.shape) != 4:
        raise ExtractionError(f"Error: Input file '{input_nifti}' is not a 4D NIfTI file. Shape: {data_4d.shape}")

    time_index = int(time_index) - 1 if time_index is not None else None
    # Check time index is valid
    max_index = data_4d.shape[3]
    if not (0 <= time_index < max_index):
        raise ExtractionError(f"Time index {time_index} is out of range. Valid range: 0 to {max_index - 1}.")

    try:
        time_index = int(time_index)
    except (ValueError, TypeError):
        raise ExtractionError(f"Invalid time index: {time_index}. Must be an integer.")

    max_index = data_4d.shape[3]
    if not (0 <= time_index < max_index):
        raise ExtractionError(f"Time index {time_index} is out of range. Valid range: 0 to {max_index - 1}.")

    # Extract the requested 3D time point
    data_3d = data_4d[:, :, :, time_index]

    if dry_run:
        print(f"Dry run: Would extract time index {time_index} from '{input_nifti}' and save to '{output_path}'.")
        return

    try:
        nii_3d = nib.Nifti1Image(data_3d, affine=nii_4d.affine, header=nii_4d.header)
        nib.save(nii_3d, output_path)
    except Exception as e:
        raise ExtractionError(f"Failed to save 3D file: {e}")

    print(f"✅ Successfully extracted time index {time_index} from '{input_nifti}' and saved to '{output_path}'.")


def main():
    """
    CLI entry point: parses arguments and runs 3D frame extraction from a 4D NIfTI file.
    """

    # Load the 4D NIfTI file

    parser = argparse.ArgumentParser(description="Extract a specific time frame (3D) from a 4D NIfTI segmentation file.")
    parser.add_argument("--input", type=str, help="Path to the 4D NIfTI segmentation file")
    parser.add_argument("--output", type=str, help="Output path for the extracted 3D NIfTI segmentation file")
    parser.add_argument("--time-index", type=int, default=None, help="If not provided, will automatically extract time index from the segmentation file name.")
    parser.add_argument("--dry-run", action="store_true", help="If set, will not perform the extraction, just print the intended output path.")
    args = parser.parse_args()
    input_nifti = Path(args.input)
    output_nifti = Path(args.output)
    if not input_nifti.exists():
        raise FileNotFoundError(f"❌ Errord: Input file '{input_nifti}' does not exist.")

    time_index = args.time_index

    try:
        extract_3d_frame(input_nifti, output_nifti, time_index, dry_run=args.dry_run)

    except Exception as e:
        print(f"❌ Error extracting 3D frame: {e}")

if __name__ == "__main__":
    main()


