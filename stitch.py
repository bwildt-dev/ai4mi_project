import argparse
from collections import defaultdict
from pathlib import Path
import nibabel as nib
import re
from tqdm import tqdm
import numpy as np
import skimage
import skimage as skimg
from concurrent.futures import ProcessPoolExecutor, as_completed, Future



def get_groups(data_path: Path, group_regex: re.Pattern) -> dict[str, list[tuple[int, Path]]]:
    groups = defaultdict(list)
    for file in data_path.iterdir():
        match = group_regex.match(file.stem)
        if match:
            patient_id = match.group(1)
            z_match = re.search(r'_(\d{4})$', file.stem)
            if not z_match:
                raise ValueError(f"Cannot extract z-index from filename: {file.name}")
            z_index = int(z_match.group(1))
        groups[patient_id].append((z_index, file))
    for patient_id in groups:
        groups[patient_id].sort(key=lambda x: x[0])
    return groups

def extract_index(name: str) -> int:
    match = re.search(r'(\d+)(?=\D*$)', name)
    return int(match.group(1)) if match else None


def stitch_patient(patient_id, patient_slices, dest_folder: Path, num_classes: int, source_pattern: str):
    dest_folder.mkdir(parents=True, exist_ok=True)

    source_path = source_pattern.format(id_=patient_id)
    gt = nib.load(source_path)
    og_size = gt.shape

    voxel = np.zeros(og_size, dtype=np.uint8)

    for z_index, file in patient_slices:
        slice_png = skimg.io.imread(file).astype(np.uint8)
        slice_png = np.clip(slice_png, 0, num_classes - 1)
        if slice_png.shape != og_size[:2]:
            slice_png = skimage.transform.resize(
                slice_png, og_size[:2], order=0, preserve_range=True, anti_aliasing=False
            ).astype(np.uint8)

        voxel[:, :, z_index] = slice_png

    stitched_img = nib.Nifti1Image(voxel, affine=gt.affine, header=gt.header)
    nib.save(stitched_img, dest_folder / f"{patient_id}_stitched.nii.gz")



def get_args():
    parser = argparse.ArgumentParser(description="Stitch sliced segmentations into full scans")
    parser.add_argument('--data_folder', type=Path, required=True, help='Folder with sliced data')
    parser.add_argument('--dest_folder', type=Path, required=True, help='Destination folder for stitched data')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--grp_regex', type=str, required=True, help='Regex pattern for grouping filenames')
    parser.add_argument('--source_scan_pattern', type=str, required=True, help='Pattern for original scans, with {id_} placeholder')
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    re_pattern = re.compile(args.grp_regex)
    groups = get_groups(args.data_folder, re_pattern)
    print(f'Found {len(groups)} patients')

    workers = min(8, len(groups)) or 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs: list[Future] = [
            ex.submit(
                stitch_patient,
                patient_id,
                slices,
                args.dest_folder,
                args.num_classes,
                args.source_scan_pattern
            )
            for patient_id, slices in groups.items()
        ]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="Stitching"):
            job.result()

if __name__ == "__main__":
    main(get_args())

