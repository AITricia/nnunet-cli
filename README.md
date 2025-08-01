# 🧠 nnU-Net Preprocessing CLI

This tool helps you prepare CT and segmentation NIfTI files for use with **nnU-Net V2**, including frame extraction and dataset folder generation.

---

## 📬 Feedback

Feel free to reach out with suggestions or questions:  
**lobotr@seas.upenn.edu**

---

## 📦 Installation

From the project root, install dependencies:

```bash
pip install .
```

To install dev dependencies (necessary for running nnUNet validations and tests):
```
pip install '.[dev]'
```
---

## 🛠️ Usage

### 1. Extract 3D frames from 4D NIfTI files

This prepares individual `.nii.gz` files per timepoint (if needed):

```
batch-extract-nifti-frames
```

Make sure your source folder structure and naming match what the script expects.

---

### 2. Generate nnU-Net-compatible dataset folder

This copies images and segmentations into the correct structure for nnU-Net v2:

```
generate-nnunet-folder-for-dataset
```

Output is organized as follows:

```
imagesTr/
labelsTr/
dataset.json
```

## 📁 Expected Input Structure:

Make sure your input files (CT and segmentation) are named like:
```
img3d_bavcta001_seg01_baseline.nii.gz
```

These filenames will be parsed to match image–label pairs.


## ✅ Output

A ready-to-train nnU-Net dataset folder:

```
nnunet_inputs/Dataset001_YourName/
├── imagesTr/
├── labelsTr/
└── dataset.json
```
This folder can be used directly with `nnUNetv2_plan_and_preprocess`.

---

## 📎 Notes

- Only the labels specified in your code will be retained in the output segmentations.
- Ensure that the label integers in your segmentation files match the definitions in your `dataset.json`.

Example:

```json
"labels": {
    "background": "0",
    "aorta": "1"
}
```


