import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic building blocks, just call this with different input and output channels accordingly to build up the network
# This is the block used for resnet18/34.
class ResidualBlock(nn.Module):

    def __init__(self, out_, strides=1, downsample=None):
        super().__init__()

        self.no_downsample = downsample is None  # Flag to check whether are we downsampling

        if self.no_downsample:
            in_ = out_  # non-downsampling layers has same channels dimension input and output
        else:
            in_ = int(out_ / 2)  # downsampling layers input channels is 2 times smaller than output channels.

        self.block = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_, out_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_),
            nn.ReLU(inplace=True)
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x  # following what the paper is calling it
        x = self.block(x)

        if not self.no_downsample:
            identity = self.downsample(identity)

        x = x + identity
        x = F.relu(x)
        return x


class ResNetBackbone(nn.Module):

    def __init__(self, channel_configs, block_configs, out_shape = (256, 256), inp_ch = 3):
        super().__init__()

        # Residual block setup: x3 64, x4 128, x6 256, x3 512 (34-layer setup as per paper)
        # First conv1 is 7x7, 64 channels, stride 2 (do not really fancy this personally)
        self.input_block = nn.Sequential(
            nn.Conv2d(inp_ch, channel_configs[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(channel_configs[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        block = ResidualBlock
        self.blocks = self._build_blocks(block, channels=channel_configs, num_blocks=block_configs)

        self.output_pooler = nn.AdaptiveAvgPool2d(out_shape)

    def forward(self, x):
        x = self.input_block(x)
        x = self.blocks(x)
        x = self.output_pooler(x)
        return x

    def _build_blocks(self, block, channels, num_blocks):

        layers = []
        downsample = None

        for i, total_block in enumerate(num_blocks):  # Go through all the blocks

            # Note only the first block no need downsample (stride=1), the rest is stride=2
            strides = 2 if i != 0 else 1
            out_ = channels[i]  # the dimension of channels for the particular block

            for _ in range(total_block):
                layers += [block(out_, strides, downsample)]
                # Only first layer in new block need downsampling (all layers that follows are back to default setting)
                strides = 1
                downsample = None

            if i == len(
                    num_blocks) - 1:  # Basically if you are at the very last block, there's no need for downsampling anymore
                break

            # After first block, next blocks (for each starting layer) will need downsampling
            downsample = nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, stride=2),
                nn.BatchNorm2d(channels[i + 1])
            )

        return nn.Sequential(*layers)


resnet18_cfg = {"block": ResidualBlock,
                "channel_configs": [64, 128, 256, 512],
                "block_configs": [2, 2, 2, 2]}

resnet34_cfg = {"block": ResidualBlock,
                "channel_configs": [64, 128, 256, 512],
                "block_configs": [3, 4, 6, 3]}
