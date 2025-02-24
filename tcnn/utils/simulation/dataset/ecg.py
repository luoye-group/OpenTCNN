import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tcnn.utils.simulation.dataset.download import download
from tcnn.utils.simulation.dataset.uncompress import unzip_file

"""
the script is used to preprocess the raw data from https://physionet.org/content/challenge-2017/1.0.0/
the implement of this script refers to https://github.com/hsd1503/resnet1d/blob/master/util.py
we remove some absolute path in the original script and others to optimize the code

Gari Clifford, Chengyu Liu, Benjamin Moody, Li-wei H. Lehman, Ikaro Silva, Qiao Li, Alistair Johnson, Roger G. Mark. 
AF Classification from a Short Single Lead ECG Recording: the PhysioNet Computing in Cardiology Challenge 2017. Computing in Cardiology (Rennes: IEEE), Vol 44, 2017 (In Press).
"""


class MyDataset(Dataset):
    """
    A custom dataset class for ECG data.

    Args:
        data (list): The input data.
        label (list): The corresponding labels.

    Returns:
        tuple: A tuple containing the input data and its corresponding label.
    """

    def __init__(self, data, label):
        """
        Initialization function

        Args:
        data (list): ECG data
        label (str): ECG label
        """
        self.data = data
        self.label = label

    def __getitem__(self, index):
        """
        Get the item at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the data and label at the specified index.
        """
        return (
            torch.tensor(self.data[index], dtype=torch.float),
            torch.tensor(self.label[index], dtype=torch.long),
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)


class ECGDataset(object):
    def __init__(self, save_dir=os.path.join(".", "data")) -> None:
        """
        Initialize the ECGDataset class.

        Args:
            save_dir (str): The directory to save the downloaded and preprocessed data. Default is './data'.

        Raises:
            AssertionError: If the preprocessed dataset 'challenge2017.pkl' is not found in the specified directory.

        """

        """
        download the raw data from https://physionet.org/content/challenge-2017/1.0.0/, 
        and put it in ../data/challenge2017/

        The preprocessed dataset challenge2017.pkl can also be found at https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf
        """
        # download dataset
        data_url = "https://physionet.org/static/published-projects/challenge-2017/af-classification-from-a-short-single-lead-ecg-recording-the-physionetcomputing-in-cardiology-challenge-2017-1.0.0.zip"
        save_path = download(data_url, save_dir)

        uncompress_dir = os.path.join(
            save_dir,
            "af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0",
        )

        if not os.path.exists(uncompress_dir):
            unzip_file(save_path, save_dir)

        # papre the preprocessed dataset challenge2017.pkl
        self.pkl_file_path = os.path.join(uncompress_dir, "challenge2017.pkl")
        pkl_download_url = (
            "https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf"
        )
        pkl_error_info = """
        Please download challenge2017.pkl from {}manually,
        and put it in {}
        """.format(
            pkl_download_url, self.pkl_file_path
        )
        assert os.path.exists(self.pkl_file_path), pkl_error_info

        # unzip training2017
        train2017_path = os.path.join(uncompress_dir, "training2017.zip")
        train2017_unzip_path = os.path.join(uncompress_dir, "training2017")
        unzip_file(train2017_path, uncompress_dir)

        # unzip sample2017
        sample2017_path = os.path.join(uncompress_dir, "sample2017.zip")
        sample2017_unzip_path = os.path.join(uncompress_dir, "sample2017")
        unzip_file(sample2017_path, uncompress_dir)

        # read label
        label_df = pd.read_csv(
            os.path.join(uncompress_dir, "REFERENCE-v3.csv"), header=None
        )
        label = label_df.iloc[:, 1].values
        print(Counter(label))

        # read data
        all_data = []
        filenames = pd.read_csv(
            os.path.join(train2017_unzip_path, "RECORDS"), header=None
        )
        filenames = filenames.iloc[:, 0].values
        print(filenames)
        fixed_length = 9000
        for filename in tqdm(filenames):
            mat = scipy.io.loadmat(
                os.path.join(train2017_unzip_path, "{0}.mat".format(filename))
            )
            mat = np.array(mat["val"])[0]
            # print(filename, mat.shape)
            all_data.append(mat)
        # all_data = np.array(all_data)

        res = {"data": all_data, "label": label}
        self.manul_pkl_file_path = os.path.join(
            uncompress_dir, "challenge2017_manual.pkl"
        )
        with open(self.manul_pkl_file_path, "wb") as fout:
            pickle.dump(res, fout)
        print("save to {}".format(self.manul_pkl_file_path))

    def slide_and_cut(self, X, Y, window_size, stride, output_pid=False, datatype=4):
        """
        Slides a window of specified size over the input data and cuts it into smaller segments.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Labels corresponding to the input data.
            window_size (int): Size of the sliding window.
            stride (int): Stride value for sliding the window.
            output_pid (bool, optional): Whether to output the corresponding patient IDs. Defaults to False.
            datatype (float or int, optional): Type of data. Defaults to 4.

        Returns:
            numpy.ndarray: Slided and cut input data.
            numpy.ndarray: Labels corresponding to the slided and cut input data.
            numpy.ndarray: Patient IDs (if output_pid is True).
        """

        out_X = []
        out_Y = []
        out_pid = []
        n_sample = X.shape[0]
        mode = 0
        for i in range(n_sample):
            tmp_ts = X[i]
            tmp_Y = Y[i]
            if tmp_Y == 0:
                i_stride = stride
            elif tmp_Y == 1:
                if datatype == 4:
                    i_stride = stride // 6
                elif datatype == 2:
                    i_stride = stride // 10
                elif datatype == 2.1:
                    i_stride = stride // 7
            elif tmp_Y == 2:
                i_stride = stride // 2
            elif tmp_Y == 3:
                i_stride = stride // 20
            for j in range(0, len(tmp_ts) - window_size, i_stride):
                out_X.append(tmp_ts[j : j + window_size])
                out_Y.append(tmp_Y)
                out_pid.append(i)
        if output_pid:
            return np.array(out_X), np.array(out_Y), np.array(out_pid)
        else:
            return np.array(out_X), np.array(out_Y)

    def read_data_physionet_2_clean_federated(
        self, m_clients, test_ratio=0.2, window_size=3000, stride=500
    ):
        """
        Reads the Physionet dataset and prepares it for federated learning.

        Args:
            m_clients (int): The number of clients to split the dataset into.
            test_ratio (float, optional): The ratio of the dataset to be used for testing. Defaults to 0.2.
            window_size (int, optional): The size of the sliding window for data segmentation. Defaults to 3000.
            stride (int, optional): The stride length for sliding window segmentation. Defaults to 500.

        Returns:
            list: A list containing the preprocessed data for each client. Each element of the list is a list containing:
                - X_train: The training data for the client.
                - X_test: The testing data for the client.
                - Y_train: The training labels for the client.
                - Y_test: The testing labels for the client.
                - pid_test: The patient IDs for the testing data.

        Note:
            - Only 'N' and 'A' labels are included in the dataset, where 'N' represents normal and 'A' represents abnormal.
            - The dataset is evenly split into m_clients pieces.
            - The data is standardized by subtracting the mean and dividing by the standard deviation.
            - The labels are encoded as 0 for 'N' and 1 for 'A'.
            - The training and testing data are randomly split using the specified test_ratio.
            - The training data is segmented into windows of size window_size with a stride of stride.
            - The testing data is segmented into windows of size window_size with a stride of stride, and the patient IDs are also returned.

        """
        """
        - only N A, no O P
        - federated dataset, evenly cut the entire dataset into m_clients pieces
        """

        # read pkl
        with open(self.pkl_file_path, "rb") as fin:
            res = pickle.load(fin)
        ## scale data
        all_data = res["data"]
        for i in range(len(all_data)):
            tmp_data = all_data[i]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            all_data[i] = (tmp_data - tmp_mean) / tmp_std
        all_data_raw = res["data"]
        all_data = []
        ## encode label
        all_label = []
        for i in range(len(res["label"])):
            if res["label"][i] == "A":
                all_label.append(1)
                all_data.append(res["data"][i])
            elif res["label"][i] == "N":
                all_label.append(0)
                all_data.append(res["data"][i])
        all_label = np.array(all_label)
        all_data = np.array(all_data)

        # split into m_clients
        shuffle_pid = np.random.permutation(len(all_label))
        m_clients_pid = np.array_split(shuffle_pid, m_clients)
        all_label_list = [all_label[i] for i in m_clients_pid]
        all_data_list = [all_data[i] for i in m_clients_pid]

        out_data = []
        for i in range(m_clients):
            print("clinet {}".format(i))
            tmp_label = all_label_list[i]
            tmp_data = all_data_list[i]

            # split train test
            X_train, X_test, Y_train, Y_test = train_test_split(
                tmp_data, tmp_label, test_size=test_ratio, random_state=0
            )

            # slide and cut
            print("before: ")
            print(Counter(Y_train), Counter(Y_test))
            X_train, Y_train = self.slide_and_cut(
                X_train, Y_train, window_size=window_size, stride=stride, datatype=2.1
            )
            X_test, Y_test, pid_test = self.slide_and_cut(
                X_test,
                Y_test,
                window_size=window_size,
                stride=stride,
                datatype=2.1,
                output_pid=True,
            )
            print("after: ")
            print(Counter(Y_train), Counter(Y_test))

            # shuffle train
            shuffle_pid = np.random.permutation(Y_train.shape[0])
            X_train = X_train[shuffle_pid]
            Y_train = Y_train[shuffle_pid]

            X_train = np.expand_dims(X_train, 1)
            X_test = np.expand_dims(X_test, 1)

            out_data.append([X_train, X_test, Y_train, Y_test, pid_test])

        return out_data

    def read_data_physionet_2_clean(self, window_size=3000, stride=500):
        """
        Reads and preprocesses the PhysioNet dataset.

        Args:
            window_size (int): The size of the sliding window for segmentation. Default is 3000.
            stride (int): The stride length for sliding window segmentation. Default is 500.

        Returns:
            tuple: A tuple containing the preprocessed training and testing data and labels, as well as the test patient IDs.
                - X_train (ndarray): Preprocessed training data.
                - X_test (ndarray): Preprocessed testing data.
                - Y_train (ndarray): Training labels.
                - Y_test (ndarray): Testing labels.
                - pid_test (ndarray): Test patient IDs.
        """
        # read pkl
        with open(self.pkl_file_path, "rb") as fin:
            res = pickle.load(fin)
        ## scale data
        all_data = res["data"]
        for i in range(len(all_data)):
            tmp_data = all_data[i]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            all_data[i] = (tmp_data - tmp_mean) / tmp_std
        all_data_raw = res["data"]
        all_data = []
        ## encode label
        all_label = []
        for i in range(len(res["label"])):
            if res["label"][i] == "A":
                all_label.append(1)
                all_data.append(res["data"][i])
            elif res["label"][i] == "N":
                all_label.append(0)
                all_data.append(res["data"][i])
        all_label = np.array(all_label)
        all_data = np.array(all_data)

        # split train test
        X_train, X_test, Y_train, Y_test = train_test_split(
            all_data, all_label, test_size=0.1, random_state=0
        )

        # slide and cut
        print("before: ")
        print(Counter(Y_train), Counter(Y_test))
        X_train, Y_train = self.slide_and_cut(
            X_train, Y_train, window_size=window_size, stride=stride, datatype=2.1
        )
        X_test, Y_test, pid_test = self.slide_and_cut(
            X_test,
            Y_test,
            window_size=window_size,
            stride=stride,
            datatype=2.1,
            output_pid=True,
        )
        print("after: ")
        print(Counter(Y_train), Counter(Y_test))

        # shuffle train
        shuffle_pid = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_pid]
        Y_train = Y_train[shuffle_pid]

        X_train = np.expand_dims(X_train, 1)
        X_test = np.expand_dims(X_test, 1)

        return X_train, X_test, Y_train, Y_test, pid_test

    def read_data_physionet_2(self, window_size=3000, stride=500):
        """
        Read and preprocess data from PhysioNet dataset.

        Args:
            window_size (int): The size of the sliding window for data segmentation. Default is 3000.
            stride (int): The stride length for sliding window. Default is 500.

        Returns:
            tuple: A tuple containing the following elements:
                - X_train (ndarray): Training data with shape (num_samples, 1, window_size).
                - X_test (ndarray): Testing data with shape (num_samples, 1, window_size).
                - Y_train (ndarray): Training labels with shape (num_samples,).
                - Y_test (ndarray): Testing labels with shape (num_samples,).
                - pid_test (ndarray): Patient IDs for testing data with shape (num_samples,).
        """

        # read pkl
        with open(self.pkl_file_path, "rb") as fin:
            res = pickle.load(fin)
        ## scale data
        all_data = res["data"]
        for i in range(len(all_data)):
            tmp_data = all_data[i]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            all_data[i] = (tmp_data - tmp_mean) / tmp_std
        all_data = res["data"]
        ## encode label
        all_label = []
        for i in res["label"]:
            if i == "A":
                all_label.append(1)
            else:
                all_label.append(0)
        all_label = np.array(all_label)

        # split train test
        X_train, X_test, Y_train, Y_test = train_test_split(
            all_data, all_label, test_size=0.1, random_state=0
        )

        # slide and cut
        print("before: ")
        print(Counter(Y_train), Counter(Y_test))
        X_train, Y_train = self.slide_and_cut(
            X_train, Y_train, window_size=window_size, stride=stride, n_class=2
        )
        X_test, Y_test, pid_test = self.slide_and_cut(
            X_test,
            Y_test,
            window_size=window_size,
            stride=stride,
            n_class=2,
            output_pid=True,
        )
        print("after: ")
        print(Counter(Y_train), Counter(Y_test))

        # shuffle train
        shuffle_pid = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_pid]
        Y_train = Y_train[shuffle_pid]

        X_train = np.expand_dims(X_train, 1)
        X_test = np.expand_dims(X_test, 1)

        return X_train, X_test, Y_train, Y_test, pid_test

    def read_data_physionet_4(self, window_size=3000, stride=500):
        """
        Reads the PhysioNet dataset and performs preprocessing steps including scaling the data and encoding the labels.
        Splits the data into training and testing sets, slides and cuts the data into windows, shuffles the training data,
        and expands the dimensions of the input data.

        Args:
            window_size (int): The size of the sliding window for the data.
            stride (int): The stride value for sliding the window.

        Returns:
            X_train (numpy.ndarray): The training data with expanded dimensions.
            X_test (numpy.ndarray): The testing data with expanded dimensions.
            Y_train (numpy.ndarray): The training labels.
            Y_test (numpy.ndarray): The testing labels.
            pid_test (numpy.ndarray): The patient IDs for the testing data.
        """

        # read pkl
        with open(self.pkl_file_path, "rb") as fin:
            res = pickle.load(fin)
        ## scale data
        all_data = res["data"]
        for i in range(len(all_data)):
            tmp_data = all_data[i]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            all_data[i] = (tmp_data - tmp_mean) / tmp_std
        ## encode label
        all_label = []
        for i in res["label"]:
            if i == "N":
                all_label.append(0)
            elif i == "A":
                all_label.append(1)
            elif i == "O":
                all_label.append(2)
            elif i == "~":
                all_label.append(3)
        all_label = np.array(all_label)

        # split train test
        X_train, X_test, Y_train, Y_test = train_test_split(
            all_data, all_label, test_size=0.1, random_state=0
        )

        # slide and cut
        print("before: ")
        print(Counter(Y_train), Counter(Y_test))
        X_train, Y_train = self.slide_and_cut(
            X_train, Y_train, window_size=window_size, stride=stride
        )
        X_test, Y_test, pid_test = self.slide_and_cut(
            X_test, Y_test, window_size=window_size, stride=stride, output_pid=True
        )
        print("after: ")
        print(Counter(Y_train), Counter(Y_test))

        # shuffle train
        shuffle_pid = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_pid]
        Y_train = Y_train[shuffle_pid]

        X_train = np.expand_dims(X_train, 1)
        X_test = np.expand_dims(X_test, 1)

        return X_train, X_test, Y_train, Y_test, pid_test

    def read_data_physionet_4_with_val(self, window_size=3000, stride=500):
        """
        Read PhysioNet dataset with 4 classes and perform data preprocessing.

        Args:
            window_size (int): The size of the sliding window for data segmentation. Default is 3000.
            stride (int): The stride length for sliding window. Default is 500.

        Returns:
            tuple: A tuple containing the following elements:
                - X_train (numpy.ndarray): Training data with shape (num_samples, 1, window_size).
                - X_val (numpy.ndarray): Validation data with shape (num_samples, 1, window_size).
                - X_test (numpy.ndarray): Test data with shape (num_samples, 1, window_size).
                - Y_train (numpy.ndarray): Training labels with shape (num_samples,).
                - Y_val (numpy.ndarray): Validation labels with shape (num_samples,).
                - Y_test (numpy.ndarray): Test labels with shape (num_samples,).
                - pid_val (numpy.ndarray): Patient IDs for validation data with shape (num_samples,).
                - pid_test (numpy.ndarray): Patient IDs for test data with shape (num_samples,).
        """

        # read pkl
        with open(self.pkl_file_path, "rb") as fin:
            res = pickle.load(fin)
        ## scale data
        all_data = res["data"]
        for i in range(len(all_data)):
            tmp_data = all_data[i]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            all_data[i] = (tmp_data - tmp_mean) / tmp_std
        ## encode label
        all_label = []
        for i in res["label"]:
            if i == "N":
                all_label.append(0)
            elif i == "A":
                all_label.append(1)
            elif i == "O":
                all_label.append(2)
            elif i == "~":
                all_label.append(3)
        all_label = np.array(all_label)

        # split train val test
        X_train, X_test, Y_train, Y_test = train_test_split(
            all_data, all_label, test_size=0.2, random_state=0
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_test, Y_test, test_size=0.5, random_state=0
        )

        # slide and cut
        print("before: ")
        print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
        X_train, Y_train = self.slide_and_cut(
            X_train, Y_train, window_size=window_size, stride=stride
        )
        X_val, Y_val, pid_val = self.slide_and_cut(
            X_val, Y_val, window_size=window_size, stride=stride, output_pid=True
        )
        X_test, Y_test, pid_test = self.slide_and_cut(
            X_test, Y_test, window_size=window_size, stride=stride, output_pid=True
        )
        print("after: ")
        print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

        # shuffle train
        shuffle_pid = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_pid]
        Y_train = Y_train[shuffle_pid]

        X_train = np.expand_dims(X_train, 1)
        X_val = np.expand_dims(X_val, 1)
        X_test = np.expand_dims(X_test, 1)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_val, pid_test

    def read_data_generated(
        self, n_samples, n_length, n_channel, n_classes, verbose=False
    ):
        """
        Generated data

        This generated data contains one noise channel class, plus unlimited number of sine channel classes which are different on frequency.

        """
        all_X = []
        all_Y = []

        # noise channel class
        X_noise = np.random.rand(n_samples, n_channel, n_length)
        Y_noise = np.array([0] * n_samples)
        all_X.append(X_noise)
        all_Y.append(Y_noise)

        # sine channel classe
        x = np.arange(n_length)
        for i_class in range(n_classes - 1):
            scale = 2**i_class
            offset_list = 2 * np.pi * np.random.rand(n_samples)
            X_sin = []
            for i_sample in range(n_samples):
                tmp_x = []
                for i_channel in range(n_channel):
                    tmp_x.append(np.sin(x / scale + 2 * np.pi * np.random.rand()))
                X_sin.append(tmp_x)
            X_sin = np.array(X_sin)
            Y_sin = np.array([i_class + 1] * n_samples)
            all_X.append(X_sin)
            all_Y.append(Y_sin)

        # combine and shuffle
        all_X = np.concatenate(all_X)
        all_Y = np.concatenate(all_Y)
        shuffle_idx = np.random.permutation(all_Y.shape[0])
        all_X = all_X[shuffle_idx]
        all_Y = all_Y[shuffle_idx]

        # random pick some and plot
        if verbose:
            for _ in np.random.permutation(all_Y.shape[0])[:10]:
                fig = plt.figure()
                plt.plot(all_X[_, 0, :])
                plt.title("Label: {0}".format(all_Y[_]))

        return all_X, all_Y
