import logging
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from tcnn.utils.simulation.dataset.mit_bih_ecg import MITBIH, MITBIHDataset
from tcnn.utils.simulation.dataset.urban_sound_8k import UrbanSound8K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def pad_sequence(batch):
    """
    Pad the sequences in a batch with zeros to make them the same length.

    Args:
        batch (list): A list of tensors representing the sequences.

    Returns:
        torch.Tensor: The padded batch of sequences.

    """
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def pad_waveform(waveform, desired_length):
    if waveform.shape[1] > desired_length:
        waveform = waveform[:, :desired_length]
    elif waveform.shape[1] < desired_length:
        padding = torch.zeros((1, desired_length - waveform.shape[1]))
        waveform = torch.cat([waveform, padding], dim=1)
    return waveform


def label_to_index(word, labels):
    """
    Convert a label word to its corresponding index.

    Args:
        word (str): The label word.

    Returns:
        torch.Tensor: The index of the label.

    """
    return torch.tensor(labels.index(word))


def collate_fn_outside(transform, labels):
    """
    Collate function for the data loader.

    Args:
        transform (callable): A function to transform the waveform.
        labels (list): A list of labels.

    Returns:
        callable: A collate function for the data loader.

    """

    def collate_fn_inside(batch):
        """
        Collate function for the data loader.

        Args:
            batch (list): A list of data tuples.

        Returns:
            tuple: A tuple containing the batched tensors and targets.

        """
        tensors, targets = [], []

        for waveform, _, label, *_ in batch:
            waveform = transform(waveform)
            tensors += [waveform]
            targets += [label_to_index(label, labels)]

        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    return collate_fn_inside


def load_speech_command(batch_size, data_dir):
    """
    Load the speech command dataset and create data loaders.

    Args:
        batch_size (int): The batch size for the data loaders.
        data_dir (str): The folder path where the dataset is located.

    Returns:
        tuple: A tuple containing the train, test, and evaluation data loaders, input shape, output shape, and labels.

    """
    os.makedirs(data_dir, exist_ok=True)

    trainset_speechcommands = torchaudio.datasets.SPEECHCOMMANDS(
        f"./{data_dir}/",
        download=True,
        subset="training",
    )
    testset_speechcommands = torchaudio.datasets.SPEECHCOMMANDS(
        f"./{data_dir}/",
        download=True,
        subset="testing",
    )
    evalset_speechcommands = torchaudio.datasets.SPEECHCOMMANDS(
        f"./{data_dir}/", download=True, subset="validation"
    )

    waveform, sample_rate, _, _, _ = trainset_speechcommands[0]
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=new_sample_rate
    )
    transformed = transform(waveform)

    labels = sorted(list(set(datapoint[2] for datapoint in trainset_speechcommands)))
    collate_fn = collate_fn_outside(transform, labels)

    train_dataloader = DataLoader(
        trainset_speechcommands,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        testset_speechcommands,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    eval_dataloader = DataLoader(
        evalset_speechcommands,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    input_shape = transformed.shape[0]
    output_shape = len(labels)
    return (
        train_dataloader,
        test_dataloader,
        eval_dataloader,
        input_shape,
        output_shape,
        labels,
    )


def load_urban_sound_8k_icassp(batch_size, split_eval=False, data_dir="./data"):
    """
    Load the UrbanSound8K dataset and extract it if necessary.
    This implementation is used for the ICASSP 2017 paper.
    Args:
        batch_size (int): The batch size for the data loaders.
        split_eval (bool): Whether to split the dataset into training, test, and evaluation sets. Defaults to False.
        data_dir (str): The folder path where the dataset is located. Defaults to './data'.

    Returns:
        train_dataloader (DataLoader): The data loader for the training set.
        test_dataloader (DataLoader): The data loader for the test set.
        eval_dataloader (DataLoader): The data loader for the evaluation set.
        input_shape (int): The shape of the input data.
        output_shape (int): The number of output classes.
        labels (list): The list of unique labels in the dataset.

    """
    urban_sound_dataset = UrbanSound8K()
    x, y = urban_sound_dataset.get_all_training_data()
    x_test, y_test = urban_sound_dataset.get_all_testing_data()

    x = np.transpose(x, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))

    if split_eval:
        x_train, x_eval, y_train, y_eval = train_test_split(
            x, y, test_size=0.2, random_state=2024
        )
    else:
        x_train, y_train = x, y

    input_shape = x_train.shape[1]
    output_shape = 10

    print("Loading UrbanSound8K dataset:")
    print(f"Train set size: {x_train.shape[0]} Training set shape: { x_train.shape}")
    print(f"Test set size: {x_test.shape[0]} Test set shape: { x_test.shape}")
    if split_eval:
        print(
            f"Validation set size: {x_eval.shape[0]} Validation set shape: { x_eval.shape}"
        )
    print("done!")
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    if split_eval:
        x_eval = torch.tensor(x_eval, dtype=torch.float32)
        y_eval = torch.tensor(y_eval, dtype=torch.long)
        evalset = torch.utils.data.TensorDataset(x_eval, y_eval)
        eval_dataloader = DataLoader(evalset, batch_size=batch_size, shuffle=True)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    if split_eval:
        return (
            train_dataloader,
            test_dataloader,
            eval_dataloader,
            input_shape,
            output_shape,
        )
    else:
        return train_dataloader, test_dataloader, input_shape, output_shape


def load_urban_sound_8k(batch_size, test_size=0.2):
    """
    Load the UrbanSound8K dataset and extract it if necessary.

    Args:
        batch_size (int): The batch size for the data loaders.
        test_size (float): The size of the test set. Defaults to 0.2.

    Returns:
        train_dataloader (DataLoader): The data loader for the training set.
        test_dataloader (DataLoader): The data loader for the test set.
        input_shape (int): The shape of the input data.
        output_shape (int): The number of output classes.

    Examples:
        >>> train_dataloader, test_dataloader, input_shape, output_shape = load_urban_sound_8k(32)

    """
    urban_sound_dataset = UrbanSound8K()
    x, y = urban_sound_dataset.get_all_data()

    x = np.transpose(x, (0, 2, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2024
    )

    input_shape = x_train.shape[1]
    output_shape = 10

    print("Loading UrbanSound8K dataset:")
    print(f"Train set size: {x_train.shape[0]} Training set shape: { x_train.shape}")
    print(f"Test set size: {x_test.shape[0]} Test set shape: { x_test.shape}")
    print("done!")
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, input_shape, output_shape


def load_yes_no(batch_size, data_dir="./data"):
    def collate_fn_yes_no(batch):
        # Find max length in the batch
        max_length = max(waveform.shape[1] for waveform, _, _ in batch)
        # Pad waveform to max length
        batch = [
            (pad_waveform(waveform, max_length), sample_rate, torch.tensor(label))
            for waveform, sample_rate, label in batch
        ]
        # Stack all waveforms, sample_rates and labels
        waveforms = torch.stack([waveform for waveform, _, _ in batch])
        labels = torch.stack([label for _, _, label in batch])
        return waveforms, labels

    dataset_yesno = torchaudio.datasets.YESNO(data_dir, download=True)

    train_size = int(0.8 * len(dataset_yesno))
    val_size = int(0.1 * len(dataset_yesno))
    test_size = len(dataset_yesno) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset_yesno, [train_size, val_size, test_size]
    )

    print("Loading YESNO dataset")
    print(f"Train set size: {len(train_set)} Training set shape: { train_set.shape}")
    print(f"Validation set size: {len(val_set)} Validation set shape: { val_set.shape}")
    print(f"Test set size: {len(test_set)} Test set shape: { test_set.shape}")
    print("done!")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_yes_no
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_yes_no
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_yes_no
    )

    input_shape = 1
    out_shape = 8
    return train_loader, test_loader, val_loader, input_shape, out_shape


def load_gtzan(batch_size, local_url, new_sample_rate=500):
    """
    Load the speech command dataset and create data loaders.

    Args:
        batch_size (int): The batch size for the data loaders.
        data_dir (str): The folder path where the dataset is located.
        new_sample_rate (int): The new sample rate for the audio files.
    Returns:
        tuple: A tuple containing the train, test, and evaluation data loaders, input shape, output shape, and labels.

    """

    train_set = torchaudio.datasets.GTZAN(
        root=local_url, download=True, subset="training"
    )
    test_set = torchaudio.datasets.GTZAN(
        root=local_url, download=True, subset="testing"
    )
    val_set = torchaudio.datasets.GTZAN(
        root=local_url, download=True, subset="validation"
    )

    waveform, sample_rate, label = train_set[0]
    print(f"waveform shape: {waveform.shape} sample_rate: {sample_rate} ")
    transform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=new_sample_rate
    )
    transformed = transform(waveform)
    print(
        f"transformed waveform shape: {transformed.shape} sample_rate: {new_sample_rate} "
    )
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    collate_fn = collate_fn_outside(transform, labels)

    print("Loading GTZAN dataset")
    print(f"Train set size: {len(train_set)} ")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    print("done!")

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    input_shape = transformed.shape[0]
    output_shape = len(labels)

    return (
        train_dataloader,
        test_dataloader,
        eval_dataloader,
        input_shape,
        output_shape,
        labels,
    )


def load_mit_bih_ecg(data_dir, batch_size=32, ratio=0.2, random_seed=2024):
    print("loading the MIT-BIH ECG dataset")
    mit_bit = MITBIH(data_dir=data_dir)
    X_train, X_test, y_train, y_test = mit_bit.load_data(
        ratio=ratio, random_seed=random_seed
    )
    print(
        "The training data size is: ",
        X_train.shape,
        "The training label is: ",
        y_train.shape,
    )
    print(
        "The testing data size is: ",
        X_test.shape,
        "The testing label is: ",
        y_test.shape,
    )
    train_dataset, test_dataset = MITBIHDataset(X_train, y_train), MITBIHDataset(
        X_test, y_test
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def load_kaggle_ecg(save_path):
    # 设置Kaggle API密钥
    api = KaggleApi()
    api.authenticate()

    # 下载数据集
    dataset_name = "shayanfazeli/heartbeat"  # 替换为您要下载的数据集名称
    dataset_path = os.path.join(save_path, "heartbeat")  # 数据集保存路径
    zip_file_path = os.path.join(dataset_path, "heartbeat.zip")  # 压缩文件保存路径
    if os.path.exists(dataset_path) or os.path.exists(zip_file_path):
        print("Data already exists.")
    else:
        api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
        print(f"Data downloaded and unzipped to {dataset_path}")
    trainset_path = os.path.join(dataset_path, "mitbih_train.csv")
    testset_path = os.path.join(dataset_path, "mitbih_test.csv")

    train = pd.read_csv(trainset_path, header=None)
    test = pd.read_csv(testset_path, header=None)

    train_len = len(train)
    test_len = len(test)
    print(f"Load ECG Dataset from Kaggle:")
    print(f"Train set size: {train_len}")
    print(f"Test set size: {test_len}")
    X_train, y_train = train.iloc[:, :187], train[187]
    X_test, y_test = test.iloc[:, :187], test[187]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(
        1
    )  # Add channel dimension
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(
        1
    )  # Add channel dimension
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader for training and testing data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset
