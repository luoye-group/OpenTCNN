import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch
import wfdb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import seed
from torch.utils.data import DataLoader, Dataset

"""
This is a dataset loader for the MIT-BIH ECG dataset.
Ref to https://github.com/lxysl/mit-bih_ecg_recognition/tree/master
we modified the code to fit the requirements of our project.
"""


class MITBIH:
    # wavelet denoise preprocess using mallat algorithm
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def denoise(self, data):
        # wavelet decomposition
        coeffs = pywt.wavedec(data=data, wavelet="db5", level=9)
        cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

        # denoise using soft threshold
        threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
        cD1.fill(0)
        cD2.fill(0)
        for i in range(1, len(coeffs) - 2):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)

        # get the denoised signal by inverse wavelet transform
        rdata = pywt.waverec(coeffs=coeffs, wavelet="db5")
        return rdata

    # load the ecg data and the corresponding labels, then denoise the data using wavelet transform
    def get_data_set(self, number, X_data, Y_data):
        from os.path import join

        ecgClassSet = ["N", "A", "V", "L", "R"]

        # load the ecg data record
        # print("loading the ecg data of No." + number)
        record = wfdb.rdrecord(join(self.data_dir, number), channel_names=["MLII"])
        data = record.p_signal.flatten()
        rdata = self.denoise(data=data)

        # get the positions of R-wave and the corresponding labels
        annotation = wfdb.rdann(join(self.data_dir, number), "atr")
        Rlocation = annotation.sample
        Rclass = annotation.symbol

        # remove the unstable data at the beginning and the end
        start = 10
        end = 5
        i = start
        j = len(annotation.symbol) - end

        # the data with specific labels (N/A/V/L/R) required in this record are selected, and the others are discarded
        # X_data: data points of length 300 around the R-wave
        # Y_data: convert N/A/V/L/R to 0/1/2/3/4 in order
        while i < j:
            try:
                lable = ecgClassSet.index(Rclass[i])
                x_train = rdata[Rlocation[i] - 99 : Rlocation[i] + 201]
                X_data.append(x_train)
                Y_data.append(lable)
                i += 1
            except ValueError:
                i += 1
        return

    def load_data(self, ratio, random_seed):
        numberSet = [
            "100",
            "101",
            "103",
            "105",
            "106",
            "107",
            "108",
            "109",
            "111",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "119",
            "121",
            "122",
            "123",
            "124",
            "200",
            "201",
            "202",
            "203",
            "205",
            "208",
            "210",
            "212",
            "213",
            "214",
            "215",
            "217",
            "219",
            "220",
            "221",
            "222",
            "223",
            "228",
            "230",
            "231",
            "232",
            "233",
            "234",
        ]
        dataSet = []
        lableSet = []
        for n in numberSet:
            self.get_data_set(n, dataSet, lableSet)

        # reshape the data and split the dataset
        dataSet = np.array(dataSet).reshape(-1, 300)
        lableSet = np.array(lableSet).reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(
            dataSet, lableSet, test_size=ratio, random_state=random_seed
        )
        return X_train, X_test, y_train, y_test


# define the dataset class
class MITBIHDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)
