#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import random
import argparse
import warnings
from pathlib import Path
import nibabel as nib
from functools import partial
from multiprocessing import Pool
from typing import Callable
import os
import skimage as skimg
import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize


from utils import map_, tqdm_


def norm_arr(img: np.ndarray) -> np.ndarray:

    # Normalizing to [0, 255], using min-max normalization.
    hu_min, hu_max = img.min(), img.max()
    png_min, png_max = np.float64(0.0), np.float64(255.0)

    norm = (img - hu_min) / (hu_max - hu_min)
    scaled = norm * (png_max - png_min) + png_min

    # If the scaled image is out of range, issue a warning.
    if scaled.min() != png_min or scaled.max() != png_max:
        warnings.warn("Scaled image is out of range")

    return scaled.astype(np.uint8)


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = (id_path / f"{id_}.nii.gz")
    gt_path: Path = (id_path / "GT_fixed.nii.gz")
    assert id_path.exists()
    assert ct_path.exists(), ct_path

    ct_img = nib.load(ct_path)

    # Load the CT image, check sanity, and normalise it.
    ct = ct_img.get_fdata().astype(np.int16)
    ct_x, ct_y, ct_z = ct.shape

    ct_dx, ct_dy, ct_dz = ct_img.header.get_zooms()[:3]

    assert sanity_ct(ct, ct_x, ct_y, ct_z, ct_dx, ct_dy, ct_dz), f"CT sanity failed for {id_}"

    ct = norm_arr(ct)

    # If not in test mode, load the GT image and check sanity.
    gt = None
    if not test_mode:
        gt_img = nib.load(gt_path)
        gt = gt_img.get_fdata().astype(np.uint8)
        assert sanity_gt(gt, ct), f"GT sanity failed for {id_}"

    img_dir = dest_path / "img"
    gt_dir = dest_path / "gt"

    img_dir.mkdir(parents=True, exist_ok=True)
    if not test_mode:
        gt_dir.mkdir(parents=True, exist_ok=True)


    # Slice the 3D volumes into 2D slices, resize to given shape, and save PNGs.
    for z_id in range(ct_z):
        z_img_slice = ct[:, :, z_id]
        z_img_slice_png = skimg.transform.resize(z_img_slice, shape, order=1,
            mode="edge",
            anti_aliasing=True,
            preserve_range=True).astype(np.uint8)

        img_file = f"{id_}_{z_id:04d}.png"
        skimg.io.imsave(img_dir / img_file, z_img_slice_png, check_contrast=False)

        if gt is not None:
            z_gt_slice = gt[:, :, z_id]
            z_gt_slice_png = skimg.transform.resize(
                z_gt_slice,
                shape,
                order=0,
                mode="edge",
                anti_aliasing=False,
                preserve_range=True
            )
            # Make labels more distinct in grayscale PNG.
            z_gt_slice_png = (z_gt_slice_png * 63).astype(np.uint8)
            assert z_gt_slice_png.dtype == np.uint8, z_gt_slice_png.dtype
            assert set(np.unique(z_gt_slice_png)) <= {0, 63, 126, 189, 252}, np.unique(z_gt_slice_png)

            gt_file = f"{id_}_{z_id:04d}.png"
            skimg.io.imsave(gt_dir / gt_file, z_gt_slice_png, check_contrast=False)

    return ct_dx, ct_dy, ct_dz


"""
TODO: Implement a simple train/val split.
Requirements:
  - List patient IDs from <src_path>/train (folder names).
  - Shuffle them (respect a seed set in main()).
  - Take the first `retains` as validation, and the rest as training.
  - Return (training_ids, validation_ids).
"""

def get_splits(src_path: Path, retains: int) -> tuple[list[str], list[str]]:
    # Remove .DS_Store if present.
    if os.path.exists(src_path / "train/.DS_Store"):
        os.remove(src_path / "train/.DS_Store")

    train_dirs: list[str] = os.listdir(src_path / "train")
    for dirs in train_dirs:
        print(dirs)

    # Shuffle the list of patient IDs, then split into training and validation sets.
    random.shuffle(train_dirs)
    val_ids: list[str] = train_dirs[:retains]
    train_ids: list[str] = train_dirs[retains:]
    return train_ids, val_ids



def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val", "test"], [training_ids, validation_ids, test_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test')
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
