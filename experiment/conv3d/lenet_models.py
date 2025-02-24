import torch
import torch.nn as nn
import torch.nn.functional as F

import tcnn.layers as tlayers


class LeNet(nn.Module):
    """
    LeNet model ref to https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html#id2
    we modify it to the implementation of PyTorch Class
    """

    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = nn.Conv3d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(F.sigmoid(self.conv1(input)), kernel_size=2, stride=2)
        x = F.avg_pool3d(F.sigmoid(self.conv2(x)), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNetReLU(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = nn.Conv3d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(F.relu(self.conv1(input)), kernel_size=2, stride=2)
        x = F.avg_pool3d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TropicalLeNet1(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.MinPlusSumConv3d(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.MaxPlusSumConv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TropicalLeNet2(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.MinPlusMaxConv3d(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.MaxPlusMinConv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TropicalLeNet3(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.MinPlusSumConv3d(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ParallelTropicalLeNet1(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ParallelMinMaxPlusSumConv3d1p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.ParallelMinMaxPlusSumConv3d1p(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ParallelTropicalLeNet2(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ParallelMinMaxPlusSumConv3d2p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.ParallelMinMaxPlusSumConv3d2p(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ParallelTropicalLeNet3(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ParallelMinMaxPlusSumConv3d1p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ParallelTropicalLeNet4(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ParallelMinMaxPlusSumConv3d2p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CompoundTropicalLeNet1(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.CompoundMinMaxPlusSumConv3d1p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.CompoundMinMaxPlusSumConv3d1p(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CompoundTropicalLeNet2(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.CompoundMinMaxPlusSumConv3d2p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.CompoundMinMaxPlusSumConv3d2p(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CompoundTropicalLeNet3(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.CompoundMinMaxPlusSumConv3d1p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CompoundTropicalLeNet4(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.CompoundMinMaxPlusSumConv3d2p(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConstantParallelTropicalLeNet1(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ConstantParallelMinMaxPlusSumConv3d(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.ConstantParallelMinMaxPlusSumConv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class ConstantParallelTropicalLeNet2(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ConstantParallelMinMaxPlusSumConv3d(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConstantCompoundTropicalLeNet1(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ConstantCompoundMinMaxPlusSumConv3d(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = tlayers.ConstantCompoundMinMaxPlusSumConv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class ConstantCompoundTropicalLeNet2(nn.Module):
    def __init__(self, input_channels, num_classes, linear_size=16 * 5 * 5):
        super().__init__()

        self.conv1 = tlayers.ConstantCompoundMinMaxPlusSumConv3d(
            input_channels, 6, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(6, 16, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):

        x = F.avg_pool3d(self.conv1(input), kernel_size=2, stride=2)
        x = F.avg_pool3d(self.conv2(x), kernel_size=2, stride=2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model_dict(input_channels, num_classes, linear_size=16 * 5 * 5):
    """
    return a dict of models
    """
    return {
        "cptcnn1": ConstantParallelTropicalLeNet1(input_channels, num_classes, linear_size),
        "cptcnn2": ConstantParallelTropicalLeNet2(input_channels, num_classes, linear_size),
        "cctcnn1": ConstantCompoundTropicalLeNet1(input_channels, num_classes, linear_size),
        "cctcnn2": ConstantCompoundTropicalLeNet2(input_channels, num_classes, linear_size),
    }

def get_model_dict_old(input_channels, num_classes, linear_size=16 * 5 * 5):
    """
    return a dict of models
    """
    return {
        "ptcnn1": ParallelTropicalLeNet1(input_channels, num_classes, linear_size),
        "ptcnn2": ParallelTropicalLeNet2(input_channels, num_classes, linear_size),
        "ptcnn3": ParallelTropicalLeNet3(input_channels, num_classes, linear_size),
        "ptcnn4": ParallelTropicalLeNet4(input_channels, num_classes, linear_size),
        "ctcnn1": CompoundTropicalLeNet1(input_channels, num_classes, linear_size),
        "ctcnn2": CompoundTropicalLeNet2(input_channels, num_classes, linear_size),
        "ctcnn3": CompoundTropicalLeNet3(input_channels, num_classes, linear_size),
        "ctcnn4": CompoundTropicalLeNet4(input_channels, num_classes, linear_size),
        "tcnn1": TropicalLeNet1(input_channels, num_classes, linear_size),
        "tcnn2": TropicalLeNet2(input_channels, num_classes, linear_size),
        "tcnn3": TropicalLeNet3(input_channels, num_classes, linear_size),
        "lenet": LeNet(input_channels, num_classes, linear_size),
        "lenet_relu": LeNetReLU(input_channels, num_classes, linear_size),
    }
