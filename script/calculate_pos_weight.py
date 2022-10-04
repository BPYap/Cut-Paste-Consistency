# Calculate the weightage of positive class given a set of binary masks

import argparse
import math
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory of masks."
    )
    args = parser.parse_args()

    mask_dir = args.mask_dir

    all_pixel_count = 0
    positive_pixel_count = 0
    for filename in tqdm(os.listdir(mask_dir)):
        path = os.path.join(mask_dir, filename)
        with Image.open(path) as mask:
            np_image = np.asarray(mask)
            height, width = np_image.shape
            all_pixel_count += height * width
            positive_pixel_count += np_image.sum()

    pos_weight = math.log(all_pixel_count / positive_pixel_count)
    print()
    print(f"positive weightage: {pos_weight:.2f}")
