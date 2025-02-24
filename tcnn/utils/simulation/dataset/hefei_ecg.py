"""
This script is used to generate the dataset for the ECG data from the Hefei dataset.
Ref to https://github.com/JavisPeng/ecg_pytorch/tree/master
"""

import copy
import os

import numpy as np
import pandas as pd
import pywt
import torch
from scipy import signal
from sklearn.preprocessing import scale
from torch.utils.data import Dataset

np.random.seed(41)


"""
Dataset
"""


def resample(sig, target_point_num=None):
    """
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    """
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def verflip(sig):
    """
    信号竖直翻转
    :param sig:
    :return:
    """
    return sig[::-1, :]


def shift(sig, interval=20):
    """
    上下平移
    :param sig:
    :return:
    """
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, target_point_num=2028)
    # # 数据增强
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)
        if np.random.randn() > 0.5:
            sig = verflip(sig)
        if np.random.randn() > 0.5:
            sig = shift(sig)
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class HeifeiECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, num_classes=55, train=True):
        super(HeifeiECGDataset, self).__init__()

        self.train_dir = os.path.join(data_path, "hf_round1_train", "train")
        self.test_dir = os.path.join(data_path, "hf_round1_testA", "testA")
        self.train_label = os.path.join(data_path, "hf_round1_label.txt")
        self.test_label = os.path.join(data_path, "hf_round1_subA.txt")
        self.arrythmia = os.path.join(data_path, "hf_round1_arrythmia.txt")
        self.num_classes = num_classes
        self.train_data = os.path.join(data_path, "train.pth")

        if not os.path.exists(self.train_data):
            name2idx = self.name2index(self.arrythmia)
            idx2name = {idx: name for name, idx in name2idx.items()}
            self.train_fn(name2idx, idx2name)
        dd = torch.load(self.train_data)
        self.train = train
        self.data = dd["train"] if train else dd["val"]
        self.idx2name = dd["idx2name"]
        self.file2idx = dd["file2idx"]
        self.wc = 1.0 / np.log(dd["wc"])

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(self.train_dir, fid)
        df = pd.read_csv(file_path, sep=" ").values
        x = transform(df, self.train)
        target = np.zeros(self.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)

    """
    data process
    """

    def name2index(self, path):
        """
        把类别名称转换为index索引
        :param path: 文件路径
        :return: 字典
        """
        list_name = []
        for line in open(path, encoding="utf-8"):
            list_name.append(line.strip())
        name2indx = {name: i for i, name in enumerate(list_name)}
        return name2indx

    def split_data(self, file2idx, val_ratio=0.1):
        """
        划分数据集,val需保证每类至少有1个样本
        :param file2idx:
        :param val_ratio:验证集占总数据的比例
        :return:训练集，验证集路径
        """
        data = set(os.listdir(self.train_dir))
        val = set()
        idx2file = [[] for _ in range(self.num_classes)]
        for file, list_idx in file2idx.items():
            for idx in list_idx:
                idx2file[idx].append(file)
        for item in idx2file:
            # print(len(item), item)
            num = int(len(item) * val_ratio)
            val = val.union(item[:num])
        train = data.difference(val)
        return list(train), list(val)

    def file2index(self, path, name2idx):
        """
        获取文件id对应的标签类别
        :param path:文件路径
        :return:文件id对应label列表的字段
        """
        file2index = dict()
        for line in open(path, encoding="utf-8"):
            arr = line.strip().split("\t")
            id = arr[0]
            labels = [name2idx[name] for name in arr[3:]]
            # print(id, labels)
            file2index[id] = labels
        return file2index

    def count_labels(self, data, file2idx):
        """
        统计每个类别的样本数
        :param data:
        :param file2idx:
        :return:
        """
        cc = [0] * self.num_classes
        for fp in data:
            for i in file2idx[fp]:
                cc[i] += 1
        return np.array(cc)

    def train_fn(self, name2idx, idx2name):
        file2idx = self.file2index(self.train_label, name2idx)
        train, val = self.split_data(file2idx)
        wc = self.count_labels(train, file2idx)
        print(wc)
        dd = {
            "train": train,
            "val": val,
            "idx2name": idx2name,
            "file2idx": file2idx,
            "wc": wc,
        }
        torch.save(dd, self.train_data)


if __name__ == "__main__":
    data_path = "/home/limingbo/projects/cTCNN/experiment/ecg/data/hefei_ecg_data"
    train_dataset = HeifeiECGDataset(data_path=data_path, train=True)
    print(train_dataset[0])
