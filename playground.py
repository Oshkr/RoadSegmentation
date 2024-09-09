import os
import torch
import numpy as np

from pathlib import Path
from PIL import Image



def main():
    
    path_img = "./data/training"
    path_mask = "./data/training"

    img_paths = [file for file in sorted(Path(path_img,"image_2").iterdir())]
    mask_paths = [file for file in sorted(Path(path_mask,"gt_image_2").iterdir()) if "road" in str(file)]

    print(f"{len(img_paths)=}")
    print(f"{len(mask_paths)=}")


    min_h = 100000
    min_w = 100000

    for i in range(len(img_paths)):

        img = np.array(Image.open(str(img_paths[i])))
        mask = np.array(Image.open(str(mask_paths[i])))
        
        assert(img.shape==mask.shape)

        if img.shape[0] <= min_h and img.shape[1] <= min_w:
            min_h = img.shape[0]
            min_w = img.shape[1]

    print(min_h,min_w)


if __name__ == "__main__":
    main()