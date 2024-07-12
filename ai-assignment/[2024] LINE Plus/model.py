import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------
# 3x3 convolution layer
# --------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding = 1

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups,
                     bias=False, dilation=dilation)

# --------------------------------------------
# 1x1 convolution layer
# --------------------------------------------
def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """ 1x1 convolution

    Input shape:
        (batch size, in_planes, height, width)
    Output shape:
        (batch size, out_planes, height, width)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, groups=groups,
                     bias=False)

# --------------------------------------------
# Basic residual block of ResNet
# --------------------------------------------
class BasicBlock(nn.Module):
    """ Implement of Residual block - Original BasicBlock architecture (https://arxiv.org/pdf/1610.02915.pdf):

    This layer creates a basic residual block of ResNet architecture, which is a original Residual Unit.
    It consists of two 3x3 convolution layers, two batch normalization layers and two ReLU layers.

    Shortcut connection options:
        If the output feature map has the same dimensions as the input feature map,
        the shortcut performs identity mapping.

        If the output feature map dimensions increase(usually doubled),
        the shortcut performs downsample to match dimensions and halve the feature map size
        by using 1x1 convolution with stride 2.

    Args:
        inplanes: dimension of input feature
        planes: dimension of ouput feature
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # When the dimensions of the output feature map increase,
        # self.conv1 and self.downsample layers performs downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

# --------------------------------------------
# Bottleneck residual block of ResNet
# --------------------------------------------
class Bottleneck(nn.Module):
    """ Implement of Residual block - Original Bottleneck architecture (https://arxiv.org/pdf/1610.02915.pdf):

    This layer creates a bottleneck residual block of ResNet architecture, which is a original Residual Unit.
    It consists of three 3x3 convolution layers, three batch normalization layers and three ReLU layers.

    Shortcut connection options:
        If the output feature map has the same dimensions as the input feature map,
        the shortcut performs identity mapping.

        If the output feature map dimensions increase(usually doubled),
        the shortcut performs downsample to match dimensions and halve the feature map size
        by using 1x1 convolution with stride 2.

    Args:
        inplanes: dimension of input feature
        planes: compressed dimension after passing conv1
        expansion: ratio to expand dimension
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # When the dimensions of the output feature map increase,
        # self.conv2 and self.downsample layers performs downsample
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

# --------------------------------------------
# ResNet Backbone
# --------------------------------------------
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
      
# --------------------------------------------
# Define ResNet models
# --------------------------------------------
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
