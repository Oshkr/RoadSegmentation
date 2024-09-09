import os
import torch
import numpy as np
import torchvision.transforms as transforms


from pathlib import Path
from PIL import Image

class KittiData(torch.utils.data.Dataset):

    def __init__(
        self,
        root_path = "/mnt/c/Users/Oskar/files/Aerointel/RoadSegmentation/data/training",
        transform = None
    ):
        super().__init__()

        self.root_path = root_path
        self.transform = transform

        self.img_paths = [file for file in sorted(Path(root_path,"image_2").iterdir())]
        self.mask_paths = [file for file in sorted(Path(root_path,"gt_image_2").iterdir())] # keep only lanes
        

        self.mask_path_road = self.find_roads()

    def find_roads(self):
        mask_path_road = []
        for file in self.mask_paths:
            if "road" in str(file):
                mask_path_road.append(file)

        return mask_path_road

    def same_size(self):

        return transform_type("None")

    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self,idx):
        
        image_path = self.img_paths[idx]
        mask_path = self.mask_path_road[idx] # keep only lanes

        identical_size,_ = self.same_size()

        image = Image.open(str(image_path))
        image = identical_size(image)
        mask = Image.open(str(mask_path))
        mask = identical_size(mask)/255

        if self.transform:
            transform_data, transform_mask = transform_type(self.transform)
            
            image = transform_data(image)
            mask = transform_mask(mask)
        
        # in mask, 3 channels - RGB:
        #   - R: opposite side of road
        #   - G: background
        #   - B: current side of road

        # For now, only keep the current side of road
        mask = mask[-1,:,:]

        return image, mask, str(mask_path)


def transform_type(type = "train"):

    down_scale = 2
    hw_ratio = int(1226/370) # 3.31
    h_init = 256

    if type == "None":
        transform_images = transforms.Compose([
            transforms.CenterCrop((370,1226)),
            transforms.PILToTensor()
        ])
        transform_mask = None
    
    if type == "basic":

        transform_images = transforms.Compose([
            # transforms.Resize((int(370/down_scale),int(1226/down_scale))),
            transforms.Resize((h_init,h_init*hw_ratio)),
            transforms.ConvertImageDtype(torch.float),
        ])

        transform_mask = transforms.Compose([
            # transforms.Resize((int(370/down_scale),int(1226/down_scale))),
            transforms.Resize((h_init,h_init*hw_ratio))
        ])

    if type == "train":
        
        transform_images = transforms.Compose([
            # transforms.Resize((int(370/down_scale),int(1226/down_scale))),
            transforms.Resize((h_init,h_init*hw_ratio)),
            transforms.ConvertImageDtype(torch.float),
        ])

        transform_mask = transforms.Compose([
            # transforms.Resize((int(370/down_scale),int(1226/down_scale))),
            transforms.Resize((h_init,h_init*hw_ratio))
        ])

    
    return transform_images,transform_mask