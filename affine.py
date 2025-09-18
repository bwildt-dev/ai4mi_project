from pathlib import Path
import os
import argparse
import nibabel as nib
from scipy import ndimage as ndi
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_patient_paths(src_path: Path) -> list[Path]:
    patient_dirs: list[str] = os.listdir(src_path)
    patient_dirs = [dirs for dirs in patient_dirs if dirs.startswith("Patient_")]

    for dirs in patient_dirs:
        print(dirs)
    print(f"Total patients found: {len(patient_dirs)}")

    patient_paths: list[Path] = []
    for patient_dir in patient_dirs:
        patient_paths.append(src_path / patient_dir)

    return patient_paths


def get_files_to_transform(src_path: Path) -> list[Path]:
    patient_dirs: list[Path] = get_patient_paths(src_path)

    files_to_transform: list[Path] = []
    for patient_dir in patient_dirs:
        files_to_transform.append(patient_dir / "GT.nii.gz")

    return files_to_transform

def get_restore_matrix() -> np.ndarray:

    t1  = np.array([[1,0,0,275],
                    [0,1,0,200],
                    [0,0,1,0],
                    [0,0,0,1]])

    phi = -(27 / 180) * np.pi

    r2 = np.array([[np.cos(phi), -np.sin(phi), 0, 0],
                    [np.sin(phi), np.cos(phi), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    t3 = np.linalg.inv(t1)

    t4 = np.array([[1, 0, 0, 50],
                    [0, 1, 0, 40],
                    [0, 0, 1, 15],
                    [0, 0, 0, 1]])

    inv_t1 = t3
    inv_r2 = r2.T
    inv_t3 = t1
    inv_t4 = np.linalg.inv(t4)

    return inv_t4 @ inv_t3 @ inv_r2 @ inv_t1



def sanity_check(to_transform: list[Path]) -> None:
    assert len(to_transform) > 0, "No files to transform found."
    for file_path in to_transform:
        assert file_path.exists(), f"File {file_path} does not exist."

def reverse_tampering(tampered_gt_paths: list[Path]) -> None:
    restore_m = get_restore_matrix()
    heart_label = 2

    print(f"Processing {len(tampered_gt_paths)} images...")

    for tampered_gt_path in tqdm(tampered_gt_paths):
        gt_tamp = nib.load(tampered_gt_path)
        gt_tamp_data = gt_tamp.get_fdata().astype(np.uint8)
        heart_seg = np.zeros_like(gt_tamp_data)
        heart_seg[gt_tamp_data == heart_label] = heart_label
        heart_restored = ndi.affine_transform(heart_seg, restore_m, order=0)

        gt_fixed = gt_tamp_data.copy()
        gt_fixed[gt_fixed == heart_label] = 0
        gt_fixed[heart_restored == heart_label] = heart_label
        gt_fixed = gt_fixed.astype(np.uint8)

        restored_gt = nib.Nifti1Image(gt_fixed, gt_tamp.affine, gt_tamp.header)
        path_to_save = Path(tampered_gt_path.parent / "GT_fixed.nii.gz")
        nib.save(restored_gt, path_to_save)




def main(args: argparse.Namespace) -> None:
    src_path: Path = Path(args.source_dir)
    gt_paths: list[Path] = get_files_to_transform(src_path)
    sanity_check(gt_paths)
    reverse_tampering(gt_paths)






def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Fix tampered GT files")

    parser.add_argument('--source_dir', type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())