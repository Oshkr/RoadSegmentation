import os
import torch
import numpy as np

from pathlib import Path
from PIL import Image



def main():
    
    path = "./data/training/image_2/."

    for i,files in enumerate(os.listdir(path)):
        
        print(files)
        image = np.array(Image.open(Path(path,files)))
        
        print(f"{image.shape=}") # 375,1242
        print(f"{np.max(image)=}") # 255
        print(f"{np.min(image)=}") # 0
        print(45*"=")

        if i==10:
            break

    


if __name__ == "__main__":
    main()