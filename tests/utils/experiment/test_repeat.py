import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tcnn.utils.experiment.repeat import repeat_experiment, reset_dataloader


# Define a dummy model class for testing
class CNN(nn.Module):
    def __init__(
        self,
        channels_num,
        k1,
        k2,
        kernel_size,
        stride_size,
        linear_input_size,
        num_class,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels_num, k1, kernel_size=kernel_size, stride=stride_size, bias=False
        )
        self.conv2 = nn.Conv2d(
            k1, k2, kernel_size=kernel_size, stride=stride_size, bias=False
        )
        self.fc1 = nn.Linear(linear_input_size, num_class)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.relu(out)

        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return out


def my_reset_dataloader(dataset, batch_size, seed, shuffle=True):
    print("use reset_dataloader function defined by ours")
    torch.manual_seed(seed)  # 设置随机数种子
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# Define a dummy dataset for testing
train_dataset = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

test_dataset = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

# Define a dummy criterion for testing
criterion = torch.nn.CrossEntropyLoss()
channels_num = 1
k1 = 4
k2 = 4
kernel_size = 3
stride_size = 2
linear_input_size = 144
num_class = 10

# Test case 1: Running 3 experiments with 2 epochs each
results, mean, var, std = repeat_experiment(
    CNN,
    [train_dataset, test_dataset],
    my_reset_dataloader,
    32,
    criterion,
    torch.optim.Adam,
    123,
    5,
    5,
    "dummy_experiment",
    channels_num,
    k1,
    k2,
    kernel_size,
    stride_size,
    linear_input_size,
    num_class,
)


# Test case 2: Running 1 experiment with 5 epochs
results, mean, var, std = repeat_experiment(
    CNN,
    [train_dataset, test_dataset],
    my_reset_dataloader,
    32,
    criterion,
    torch.optim.Adam,
    123,
    5,
    5,
    "dummy_experiment",
    channels_num,
    k1,
    k2,
    kernel_size,
    stride_size,
    linear_input_size,
    num_class,
)
