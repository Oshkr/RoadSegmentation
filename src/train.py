# https://github.com/asujaykk/Road-segmentation-UNET-model/tree/main
# https://github.com/aschneuw/road-segmentation-unet
# https://github.com/zhechen/PLARD/tree/master

import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from torchvision.utils import save_image
from monai.losses import FocalLoss, DiceLoss, DiceCELoss

from data.dataloader import KittiData
from models.RoadUNet import RoadUNet


def one_epoch(epoch, train_loader, optimizer, model, criterion, device):
    ...


def main(
    device = "cpu",
    transform = None,
    batch_size = 16,
    learning_rate = 1e-3,
    weight_decay = 0,
    loss_name = "focal",
    epochs = 2,
    prob = 0.1
):
    
    dataset = KittiData(transform = transform)
    train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False)

    model = RoadUNet()
    model.to(device)

    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    if loss_name == "BCE":
        criterion = BCEWithLogitsLoss(reduction = "mean")
    else:
        criterion = FocalLoss(reduction = "mean")
    
    
    step_loss = []

    for epoch in tqdm(range(epochs)):

        epoch_loss = 0
        step = 0
        for i,data in enumerate(train_loader):
            
            img,mask,path = data
            img = img.to(device)
            mask = mask.to(device).unsqueeze(1)
            mask = torch.round(mask)

            # print(f"{mask.shape=}")
            # print(f"{torch.unique(mask)=}")
            # print(f"{len(torch.unique(mask))=}")

            # print(f"{out.shape=}")
            # print(f"{torch.unique(out)=}")
            # print(f"{len(torch.unique(out))=}")

            optimizer.zero_grad()
            out = model(img)

            try:
                assert mask.shape == out.shape
            except:
                print(f"{mask.shape=}")
                print(f"{out.shape=}")

            loss = criterion(out,mask)

            # everywhere where prob>0.5, gets a label
            # out = (out>prob).float()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            step += 1

            # Very first loss, without any training
            if i == 0 and epoch == 0:
                step_loss.append(loss.detach().item())
        
        step_loss.append(epoch_loss / step)

        print(f"{step_loss[epoch+1]=}")

    print(f"All Losses: {step_loss=}")


    test_img = img[0].detach().cpu()
    test_mask = mask[0].detach().cpu()
    test_out = out[0].detach().cpu()

    print(f"{torch.unique(test_img)=}")

    print(45*"=")
    print(f"{test_img.shape=}")
    print(f"{test_mask.shape=}")
    print(f"{test_out.shape=}")

    save_image(test_img, f"./test/Unet/epoch{epochs}_p{prob}_{loss_name}_test_img_nosmooth.png")
    save_image(test_mask, f"./test/Unet/epoch{epochs}_p{prob}_{loss_name}_test_mask_nosmooth.png")
    save_image(test_out, f"./test/Unet/epoch{epochs}_p{prob}_{loss_name}_test_out_nosmooth.png")

    # metric = IOU

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running with {device=}")

    seed = 123

    if seed:
        set_seed = seed
        torch.manual_seed(set_seed)
    else:
        set_seed = "no"
    print(f'Seeding: {set_seed}.')


    batch_size = 32
    transform = "basic"
    learning_rate = 3e-3
    weight_decay = 1e-3
    
    loss_name = "BCE"
    epochs = 2
    prob = 0.7

    main(device, transform, batch_size, learning_rate, weight_decay, loss_name, epochs, prob)

    torch.cuda.empty_cache()