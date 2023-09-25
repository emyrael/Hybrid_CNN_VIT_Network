import torch
import torch.nn as nn
import torch.nn.functional as F

from .ResnetBackbone import ResNetBackbone
from .VIT import VIT
from .gradcam import GradCam

from typing import List


class VitConvNet(nn.Module):

    def __init__(self, backbone_cfg, vit_cfg):
        super().__init__()

        assert vit_cfg["img_size"] == backbone_cfg["out_shape"], \
            "Backbone output should have shape the same as input shape of VIT layer!"

        self.backbone = ResNetBackbone(**backbone_cfg)
        self.transformer = VIT(**vit_cfg)

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        x = self.transformer(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def set_backbone(self, net):
        self.backbone = net

    def reset_backbone(self):
        self.backbone = None

    def compute_gradcam(self, img, layer: List[str]):
        if self.backbone is not None:
            gradcam = GradCam(self, layer)
        gradcam_image = gradcam.generate_gradcam(img)

        return gradcam_image