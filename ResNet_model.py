import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResidualBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._layer(block, layers[3], out_channels=512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4),
            )

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

"""
Blocks define how many residual blocks are stacked in each of the 4 ResNet stages.

# | Model | Blocks |
# | ResNet - 18 | [2, 2, 2, 2] |
# | ResNet - 34 | [3, 4, 6, 3] |
# | ResNet - 50 | [3, 4, 6, 3] |
# | ResNet - 101 | [3, 4, 23, 3] |
# | ResNet - 152 | [3, 8, 36, 3] |
"""

def ResNet50(img_channels=3, num_classes=4):
    return ResNet(ResidualBlock, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=4):
    return ResNet(ResidualBlock, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=4):
    return ResNet(ResidualBlock, [3, 8, 36, 3], img_channels, num_classes)

"""
# Dimensions flow example: Height, Width, Channels
# Input: 224×224×3
# Conv + Pool: 56×56×64
# Layer1: 56×56×256
# Layer2: 28×28×512
# Layer3: 14×14×1024
# Layer4: 7×7×2048
# AvgPool: 1×1×2048
# FC: num_classes
"""

if __name__ == '__main__':
    # testing
    model = ResNet101()
    model = model.to('cuda')
    x = torch.randn(2, 3, 224, 224)
    x = x.to('cuda')
    y = model(x)
    print(y.shape)

