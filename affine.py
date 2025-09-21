from argparse import ArgumentParser
from pathlib import Path
import os
import argparse
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage
from scipy import ndimage as ndi
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, Future


def get_patient_paths(src_path: Path) -> list[Path]:
    """
    Function that retrieves all Patient Folders from given path and returns them in a list.
    :param src_path: Path to the folder containing the Patient folders.
    :return: List of paths to Patient folders.
    """

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
    """
    Function that retrieves all GT files from given path and returns them in a list.
    :param src_path: Path to the data folder.
    :return: List of paths to altered GT files.
    """
    patient_dirs: list[Path] = get_patient_paths(src_path)

    files_to_transform: list[Path] = []
    for patient_dir in patient_dirs:
        files_to_transform.append(patient_dir / "GT.nii.gz")

    return files_to_transform

def get_restore_matrix() -> np.ndarray:
    """
    Function that composes the matrix needed to reverse the transformations initially applied to the altered segmentations.
    :return: Numpy array containing the reverse transformation.
    """

    t1: np.ndarray = np.array([[1,0,0,275],
                    [0,1,0,200],
                    [0,0,1,0],
                    [0,0,0,1]])

    phi: float = -(27 / 180) * np.pi
    cos: float = np.cos(phi)
    sin: float = np.sin(phi)

    r2: np.ndarray = np.array([[cos, -cos, 0, 0],
                    [sin, cos, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    t3: np.ndarray = np.linalg.inv(t1)

    t4: np.ndarray = np.array([[1, 0, 0, 50],
                    [0, 1, 0, 40],
                    [0, 0, 1, 15],
                    [0, 0, 0, 1]])

    inv_t1 = t3
    inv_r2: np.ndarray = r2.T
    inv_t3: np.ndarray = t1
    inv_t4: np.ndarray = np.linalg.inv(t4)

    return inv_t4 @ inv_t3 @ inv_r2 @ inv_t1



def sanity_check(to_transform: list[Path]) -> None:
    """
    Function that performs a few sanity checks the altered GT files before restoring them.
    :param to_transform: list of paths to altered GT files.
    """
    assert len(to_transform) > 0, "No files to transform found."
    for file_path in to_transform:
        assert file_path.exists(), f"File {file_path} does not exist."

def restore(tampered_gt_path: Path, restore_m: np.ndarray) -> None:
    """
    Function that restores the altered heart segmentations in the given GT file and saves them with _fixed suffix.
    :param tampered_gt_path: Path to the GT file to be restored.
    :param restore_m: Numpy array containing the reverse transformation.
    """
    heart_label = 2

    gt_tamp: FileBasedImage = nib.load(tampered_gt_path)
    gt_tamp_data: np.ndarray = gt_tamp.get_fdata().astype(np.uint8)
    heart_seg: np.ndarray = np.zeros_like(gt_tamp_data)
    heart_seg[gt_tamp_data == heart_label] = heart_label
    heart_restored = ndi.affine_transform(heart_seg, restore_m, order=0)

    gt_fixed: np.ndarray = gt_tamp_data.copy()
    gt_fixed[gt_fixed == heart_label] = 0
    gt_fixed[heart_restored == heart_label] = heart_label
    gt_fixed = gt_fixed.astype(np.uint8)

    restored_gt: FileBasedImage = nib.Nifti1Image(gt_fixed, gt_tamp.affine, gt_tamp.header)
    path_to_save: Path = Path(tampered_gt_path.parent / "GT_fixed.nii.gz")
    nib.save(restored_gt, path_to_save)


def parallel_restore(tampered_gt_paths: list[Path], workers: int | None = None) -> None:
    """
    Function that runs restoration operations in parallel, amount of workers depends on the number of CPU cores.
    :param tampered_gt_paths:
    :param workers:
    """
    if workers is None:
        workers: int  = min(os.cpu_count(), 12) or 1
    restore_m: np.ndarray = get_restore_matrix()
    print(f"Restoring {len(tampered_gt_paths)} images with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        jobs: list[Future] = [ex.submit(restore, path, restore_m) for path in tampered_gt_paths]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="Restoring"):
            job.result()


def main(args: argparse.Namespace) -> None:
    """
    Function that handles main orchestration of the script.
    :param args: Namespace containing command line arguments.
    """
    src_path: Path = Path(args.source_dir)
    gt_paths: list[Path] = get_files_to_transform(src_path)
    sanity_check(gt_paths)
    parallel_restore(gt_paths, 8)




def get_args() -> argparse.Namespace:
    """
    Function that parses command line arguments and returns them.
    :return: Namespace containing command line arguments.
    """
    parser: ArgumentParser = argparse.ArgumentParser(description = "Fix tampered GT files")

    parser.add_argument('--source_dir', type=str, required=True)

    args: argparse.Namespace = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())