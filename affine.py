from pathlib import Path
import os
import argparse
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
import mlx.core as mx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import math

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

def get_restore_matrix() -> mx.array:

    t1  = mx.array([[1., 0., 0., 275.],
                    [0., 1., 0., 200.],
                    [0., 0., 1.,   0.],
                    [0., 0., 0.,   1.]])

    t4 = mx.array([[1., 0., 0.,  50.],
                   [0., 1., 0.,  40.],
                   [0., 0., 1.,  15.],
                   [0., 0., 0.,   1.]])


    phi = -27.0 * math.pi / 180.0
    c, s = math.cos(phi), math.sin(phi)
    r2 = mx.array([[ c, -s, 0., 0.],
                   [ s,  c, 0., 0.],
                   [0.,  0., 1., 0.],
                   [0.,  0., 0., 1.]])


    inv_t1 = mx.array([[1., 0., 0., -275.],
                       [0., 1., 0., -200.],
                       [0., 0., 1.,    0.],
                       [0., 0., 0.,    1.]])

    inv_r2 = mx.transpose(r2)

    inv_t4 = mx.array([[1., 0., 0., -50.],
                       [0., 1., 0., -40.],
                       [0., 0., 1., -15.],
                       [0., 0., 0.,   1.]])

    inv_t3 = t1

    return inv_t4 @ inv_t3 @ inv_r2 @ inv_t1



def sanity_check(to_transform: list[Path]) -> None:
    assert len(to_transform) > 0, "No files to transform found."
    for file_path in to_transform:
        assert file_path.exists(), f"File {file_path} does not exist."

def restore(tampered_gt_path: Path, restore_m: mx.array) -> None:
    heart_label = 2

    gt_tamp = nib.load(tampered_gt_path)
    gt_tamp_data = mx.array(gt_tamp.get_fdata().astype('uint8'))
    heart_seg = mx.zeros_like(gt_tamp_data)
    heart_seg = mx.where(gt_tamp_data == heart_label, heart_label, 0)
    heart_seg_np = np.array(heart_seg)
    restore_m_np = np.array(restore_m)
    heart_restored_np = ndi.affine_transform(heart_seg_np, restore_m_np, order=0)
    heart_restored = mx.array(heart_restored_np)

    gt_fixed = mx.array(gt_tamp_data)
    gt_fixed = mx.where(gt_fixed == heart_label, 0, gt_fixed)
    gt_fixed = mx.where(heart_restored == heart_label, heart_label, gt_fixed)
    gt_fixed = np.array(gt_fixed, copy=True).astype("uint8")

    restored_gt = nib.Nifti1Image(gt_fixed, gt_tamp.affine, gt_tamp.header)
    path_to_save = Path(tampered_gt_path.parent / "GT_fixed.nii.gz")
    nib.save(restored_gt, path_to_save)


def parallel_restore(tampered_gt_paths: list[Path], workers: int | None = None) -> None:
    if workers is None:
        workers = min(os.cpu_count(), 12) or 1
    restore_m = get_restore_matrix()
    print(f"Restoring {len(tampered_gt_paths)} images with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = [ex.submit(restore, path, restore_m) for path in tampered_gt_paths]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="Restoring"):
            job.result()


def main(args: argparse.Namespace) -> None:
    src_path: Path = Path(args.source_dir)
    gt_paths: list[Path] = get_files_to_transform(src_path)
    sanity_check(gt_paths)
    parallel_restore(gt_paths, 8)




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Fix tampered GT files")

    parser.add_argument('--source_dir', type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())
