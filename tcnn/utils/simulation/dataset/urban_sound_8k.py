import logging
import os
import pickle
import tarfile
from glob import glob, iglob
from pathlib import Path
from random import choice
from shutil import rmtree
from time import time
from tkinter import Y
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

"""
This script is used to load the UrbanSound8K dataset.
The implementation ref to:
- paper: https://arxiv.org/abs/1610.00087
- GitHub: https://github.com/philipperemy/very-deep-convnets-raw-waveforms
"""


def mkdir_p(path):
    import errno

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def del_folder(path):
    try:
        rmtree(path)
    except:
        pass


def read_audio_from_filename(filename, target_sr):
    audio, _ = librosa.load(filename, sr=target_sr, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


class UrbanSound8K(torch.utils.data.Dataset):
    def __init__(self, data_dir="./data") -> None:
        """
        Initialize the UrbanSound8K dataset.

        Args:
            data_dir (str, optional): The directory where the dataset is stored. Defaults to './data'.
        """
        self._dataset_name = "UrbanSound8K.tar.gz"
        self._data_dir = data_dir
        self._tar_file_path = os.path.join(self._data_dir, self._dataset_name)
        self._url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"

        warning_message = f"Please download the dataset from {self._url} to {os.path.abspath(self._data_dir)}"
        assert os.path.exists(self._tar_file_path), warning_message

        tar = tarfile.open(self._tar_file_path)
        self.dataset_path = os.path.join(data_dir, "UrbanSound8K")

        if not os.path.exists(self.dataset_path):
            logging.info(f"exact {self._tar_file_path} to {data_dir}")
            tar.extractall(data_dir)
        else:
            logging.info(f"{self.dataset_path} exists!")

        self._dataset_audio_path = os.path.join(self.dataset_path, "audio")
        self._meta_path = os.path.join(
            self.dataset_path, "metadata", "UrbanSound8K.csv"
        )
        self._meta_info = pd.read_csv(self._meta_path)

        self._target_sr = 8000
        self._audio_length = 32000
        self._output_dir = os.path.join(data_dir, "converted_audio")
        self._output_dir_train = os.path.join(self._output_dir, "train")
        self._output_dir_test = os.path.join(self._output_dir, "test")

        del_folder(self._output_dir_train)
        del_folder(self._output_dir_test)
        mkdir_p(self._output_dir_train)
        mkdir_p(self._output_dir_test)
        self.convert_data()

        self.train_files = glob(os.path.join(self._output_dir_train, "**.pkl"))
        # print('training files =', len(self.train_files))
        self.test_files = glob(os.path.join(self._output_dir_train, "**.pkl"))
        # print('testing files =', len(self.test_files))

    def next_batch_train(self, batch_size):
        return UrbanSound8K._next_batch(batch_size, self.train_files)

    def next_batch_test(self, batch_size):
        return UrbanSound8K._next_batch(batch_size, self.test_files)

    def train_files_count(self):
        return len(self.train_files)

    def test_files_count(self):
        return len(self.test_files)

    def get_all_training_data(self):
        return UrbanSound8K._get_data(self.train_files)

    def get_all_testing_data(self):
        return UrbanSound8K._get_data(self.test_files)

    def get_all_data(self):
        return UrbanSound8K._get_data(self.train_files + self.test_files)

    @staticmethod
    def _get_data(file_list, progress_bar=False):
        def load_into(_filename, _x, _y):
            with open(_filename, "rb") as f:
                audio_element = pickle.load(f)
                _x.append(audio_element["audio"])
                _y.append(int(audio_element["class_id"]))

        x, y = [], []
        for filename in file_list:
            load_into(filename, x, y)
        return np.array(x), np.array(y)

    @staticmethod
    def _next_batch(batch_size, file_list):
        return UrbanSound8K._get_data([choice(file_list) for _ in range(batch_size)])

    def extract_class_id(self, wav_filename):
        """
        The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:
        [fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken
        [classID] = a numeric identifier of the sound class (see description of classID below for further details)
        [occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording
        [sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence
        """
        return wav_filename.split("-")[1]

    def convert_data(self):
        for i, wav_filename in enumerate(
            iglob(os.path.join(self._dataset_audio_path, "**/**.wav"), recursive=True)
        ):
            class_id = self.extract_class_id(wav_filename)
            audio_buf = read_audio_from_filename(
                wav_filename, target_sr=self._target_sr
            )
            # normalize mean 0, variance 1
            audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
            original_length = len(audio_buf)
            print(
                i,
                wav_filename,
                original_length,
                np.round(np.mean(audio_buf), 4),
                np.std(audio_buf),
            )
            if original_length < self._audio_length:
                audio_buf = np.concatenate(
                    (
                        audio_buf,
                        np.zeros(shape=(self._audio_length - original_length, 1)),
                    )
                )
                print("PAD New length =", len(audio_buf))
            elif original_length > self._audio_length:
                audio_buf = audio_buf[0 : self._audio_length]
                print("CUT New length =", len(audio_buf))

            output_folder = self._output_dir_train
            if "fold10" in wav_filename:
                output_folder = self._output_dir_test
            output_filename = os.path.join(output_folder, str(i) + ".pkl")

            out = {"class_id": class_id, "audio": audio_buf, "sr": self._target_sr}
            with open(output_filename, "wb") as w:
                pickle.dump(out, w)

    def next_batch_blank(self, batch_size):
        return np.zeros(shape=(batch_size, self, 1), dtype=np.float32), np.ones(
            shape=batch_size
        )

    def get_id2label(self):
        return {
            0: "air conditioner",
            1: "car horn",
            2: "children playing",
            3: "dog bark",
            4: "drilling",
            5: "engine idling",
            6: "gun shot",
            7: "jackhammer",
            8: "siren",
            9: "street music",
        }


if __name__ == "__main__":
    data_reader = UrbanSound8K()
    x_tr, y_tr = data_reader.get_all_training_data()
    x_te, y_te = data_reader.get_all_testing_data()
    x, y = data_reader.get_all_data()

    print("x_tr.shape =", x_tr.shape)
    print("y_tr.shape =", y_tr.shape)
    print("x_te.shape =", x_te.shape)
    print("y_te.shape =", y_te.shape)
    print("x.shape =", x.shape)
    print("y.shape =", y.shape)
    print("type", type(x_tr))
