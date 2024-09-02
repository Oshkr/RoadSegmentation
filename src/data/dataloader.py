import os

import torch
import numpy as np


from .utils import test

class KittiData(torch.utils.data.Dataset):

    def __init__(
        self,
        root_path = "./data/training/image_2/",
        transform = None
    ):
        super().__init__()

        self.root_path = root_path
        self.transform = transform

    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self):
        
        img = None
        label = None


        return img, label
