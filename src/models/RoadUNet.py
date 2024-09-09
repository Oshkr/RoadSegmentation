
import torch.nn as nn
from monai.networks.nets import UNet


class RoadUNet(nn.Module):


    def __init__(
        self,
        channels = (16,32,64)
    ):
        super().__init__()

        self.channels = channels

        self.model = UNet(
            spatial_dims = 2,
            in_channels = 3,
            out_channels = 1,
            channels = self.channels,
            strides=(2,) * (len(self.channels) - 1),
            kernel_size = 3,
            up_kernel_size = 3,
            dropout = 0.1,
            bias = True,
            adn_ordering = "NDA"
        )

    def forward(self,x):
        x = self.model(x)
        return x