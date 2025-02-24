import medmnist
import numpy as np
import torch
import torch.utils.data as data
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):

        if self.mul == "0.5":
            voxel = voxel * 0.5
        elif self.mul == "random":
            voxel = voxel * np.random.uniform()

        return voxel.astype(np.float32)


def collect_fn_multiclass(batch):

    data, target = zip(*batch)
    target = torch.stack([torch.squeeze(torch.tensor(t)) for t in target])
    data = torch.stack([torch.Tensor(d) for d in data])
    return data, target


def collect_fn_binary(batch):
    # 获取批次中的数据和标签
    data, target = zip(*batch)

    # 将标签从one-hot向量转换为样本标签
    target = torch.stack([torch.tensor(t) for t in target]).float()
    data = torch.stack([torch.Tensor(d) for d in data])
    return data, target


def get_dataset(data_flag, batch_size, size, task, shape_transform=False):
    download = True
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    # load the data
    # size = 28, 64, 128
    train_transform = Transform3D(mul="random") if shape_transform else Transform3D()
    eval_transform = Transform3D(mul="0.5") if shape_transform else Transform3D()

    train_dataset = DataClass(
        split="train", download=download, size=size, transform=train_transform
    )
    test_dataset = DataClass(
        split="test", download=download, size=size, transform=eval_transform
    )
    return train_dataset, test_dataset


def get_dataloader(data_flag, batch_size, size, task, shape_transform=False):
    train_dataset, test_dataset = get_dataset(
        data_flag, batch_size, size, task, shape_transform
    )
    # encapsulate data into dataloader form
    if task == "multiclass":
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collect_fn_multiclass,
        )
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collect_fn_multiclass,
        )
    elif task == "binary":
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collect_fn_binary,
        )
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collect_fn_binary,
        )
    else:
        raise ValueError("task should be either 'multiclass' or 'binary'")
    return train_loader, test_loader


def get_reset_dataloader(task):
    collect_fn = None
    if task == "multiclass":
        collect_fn = collect_fn_multiclass
    elif task == "binary":
        collect_fn = collect_fn_binary
    else:
        raise ValueError("task should be either 'multiclass' or 'binary'")

    def reset_dataloader(dataset, batch_size, shuffle=True):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collect_fn
        )
        return dataloader

    return reset_dataloader
