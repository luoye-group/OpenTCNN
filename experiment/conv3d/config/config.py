import torch

dataset2task = {
    "organmnist3d": {
        "type": "multiclass",
        "num_classes": 11,
        "order": "01",
    },
    "nodulemnist3d": {
        "type": "binary",
        "num_classes": 2,
        "order": "02",
    },
    "adrenalmnist3d": {
        "type": "binary",
        "num_classes": 2,
        "order": "03",
    },
    "fracturemnist3d": {
        "type": "multiclass",
        "num_classes": 3,
        "order": "04",
    },
    "vesselmnist3d": {
        "type": "binary",
        "num_classes": 2,
        "order": "05",
    },
    "synapsemnist3d": {
        "type": "binary",
        "num_classes": 2,
        "order": "06",
    },
}


def load_config28(dataset_name):
    config = {
        "task": dataset2task[dataset_name]["type"],
        "order": dataset2task[dataset_name]["order"],
        "data": {
            "batch_size": 32,
            "size": 28,
        },
        "network": {
            "input_channels": 1,
            "linear_size": 2000,
            "num_classes": dataset2task[dataset_name]["num_classes"],
        },
        "train": {
            "criterion": (
                torch.nn.CrossEntropyLoss()
                if dataset2task[dataset_name]["type"] == "multiclass"
                else torch.nn.BCELoss()
            ),
            "checkpoint_save_dir": "checkpoints",
            "epochs": 100,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "repeat": {
            "num_experiments": 5,
            "epochs_per_experiemnt": 100,
            "log_save_dir": "logs",
        },
    }
    return config


def load_config64(dataset_name):
    config = {
        "task": dataset2task[dataset_name]["type"],
        "order": dataset2task[dataset_name]["order"],
        "data": {
            "batch_size": 2,
            "size": 64,
        },
        "network": {
            "input_channels": 1,
            "linear_size": 43904,
            "num_classes": dataset2task[dataset_name]["num_classes"],
        },
        "train": {
            "criterion": (
                torch.nn.CrossEntropyLoss()
                if dataset2task[dataset_name]["type"] == "multiclass"
                else torch.nn.BCELoss()
            ),
            "checkpoint_save_dir": "checkpoints",
            "epochs": 100,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "repeat": {
            "num_experiments": 5,
            "epochs_per_experiemnt": 100,
            "log_save_dir": "logs",
        },
    }
    return config
