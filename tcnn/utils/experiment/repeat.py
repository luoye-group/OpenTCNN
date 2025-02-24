from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

import tcnn.layers as tlayer
from tcnn.utils.experiment.train import eval_model, train_and_test_model


def weight_reset(model):
    """
    Reset the weights of a model

    Args:
        model (torch.nn.Module): The model to reset the weights of.
    Returns:
        None
    Examples:
        # Reset the weights of a model
        model.apply(weight_reset)
    """
    if (
        isinstance(model, torch.nn.Conv1d)
        or isinstance(model, torch.nn.Conv2d)
        or isinstance(model, torch.nn.Conv3d)
        or isinstance(model, torch.nn.Linear)
        or isinstance(model, tlayer.Conv1d)
        or isinstance(model, tlayer.Conv2d)
        or isinstance(model, tlayer.Conv3d)
        or isinstance(model, tlayer.MaxPlusMaxConv1d)
        or isinstance(model, tlayer.MaxPlusMaxConv2d)
        or isinstance(model, tlayer.MaxPlusMaxConv3d)
        or isinstance(model, tlayer.MaxPlusMinConv1d)
        or isinstance(model, tlayer.MaxPlusMinConv2d)
        or isinstance(model, tlayer.MaxPlusMinConv3d)
        or isinstance(model, tlayer.MaxPlusSumConv1d)
        or isinstance(model, tlayer.MaxPlusSumConv2d)
        or isinstance(model, tlayer.MaxPlusSumConv3d)
        or isinstance(model, tlayer.MinPlusMaxConv1d)
        or isinstance(model, tlayer.MinPlusMaxConv2d)
        or isinstance(model, tlayer.MinPlusMaxConv3d)
        or isinstance(model, tlayer.MinPlusMinConv1d)
        or isinstance(model, tlayer.MinPlusMinConv2d)
        or isinstance(model, tlayer.MinPlusMinConv3d)
        or isinstance(model, tlayer.MinPlusSumConv1d)
        or isinstance(model, tlayer.MinPlusSumConv2d)
        or isinstance(model, tlayer.MinPlusSumConv3d)
        or isinstance(model, tlayer.ParallelMinMaxPlusSumConv1d1p)
        or isinstance(model, tlayer.ParallelMinMaxPlusSumConv1d2p)
        or isinstance(model, tlayer.ParallelMinMaxPlusSumConv2d1p)
        or isinstance(model, tlayer.ParallelMinMaxPlusSumConv2d2p)
        or isinstance(model, tlayer.ParallelMinMaxPlusSumConv3d1p)
        or isinstance(model, tlayer.ParallelMinMaxPlusSumConv3d2p)
        or isinstance(model, tlayer.CompoundMinMaxPlusSumConv1d1p)
        or isinstance(model, tlayer.CompoundMinMaxPlusSumConv1d2p)
        or isinstance(model, tlayer.CompoundMinMaxPlusSumConv2d1p)
        or isinstance(model, tlayer.CompoundMinMaxPlusSumConv2d2p)
        or isinstance(model, tlayer.CompoundMinMaxPlusSumConv3d1p)
        or isinstance(model, tlayer.CompoundMinMaxPlusSumConv3d2p)
    ):
        model.reset_parameters()


def reset_dataloader(dataset, batch_size, shuffle=True):
    """
    Reset the dataloader for a dataset

    Args:
        dataset (torch.utils.data.Dataset): The dataset to reset the dataloader for.
        batch_size (int): The batch size for the dataloader.
        seed (int): The seed for the random number generator.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    Returns:
        torch.utils.data.DataLoader: The dataloader for the dataset.

    Examples:
        # Reset the dataloader for a dataset
        dataloader = reset_dataloader(dataset, batch_size, seed, shuffle=True)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def initialize_model(model_class, *args, **kwargs):
    """
    Initialize a model

    Args:
        model_class (torch.nn.Module): The class of the model to initialize.
        *args: The arguments for the model class.
        **kwargs: The keyword arguments for the model class.
    Returns:
        torch.nn.Module: The initialized model.

    Examples:
        # Initialize a model
        model = initialize_model(MyModel, num_classes=10)
    """
    model = model_class(*args, **kwargs)
    model.apply(weight_reset)
    return model


def repeat_experiment(
    model,
    dataset,
    reset_dataloader,
    batch_size,
    criterion,
    def_optimizer,
    def_scheduler,
    num_experiments,
    epochs_per_experiemnt,
    experiment_name="experiment",
    checkpoint_save=False,
    checkpoint_save_dir=None,
    task="multiclass",
    scheduler_sign=None,
    *model_args,
    **model_kwargs,
):
    """
    Run multiple experiments and return the results

    Args:
        model_class (torch.nn.Module): The class of the model to use.
        dataset (tuple): A tuple containing the training and testing datasets.
        reset_dataloader (function): The function to reset the dataloader.
        batch_size (int): The batch size for the dataloader.
        criterion (torch.nn.Module): The loss function for the model.
        optimizer_class (torch.optim.Optimizer): The class of the optimizer to use.
        num_experiments (int): The number of experiments to run.
        epochs_per_experiemnt (int): The number of epochs per experiment.
        experiment_name (str, optional): The name of the experiment. Defaults to "experiment".
        *model_args: The arguments for the model class.
        **model_kwargs: The keyword arguments for the model class.

    Returns:
        tuple: A tuple containing the results, mean, variance, and standard deviation of the experiments.

    Examples:
        # Run multiple experiments
        results = repeat_experiment(
            MyModel,
            (train_dataset, test_dataset),
            reset_dataloader,
            batch_size=32,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            seed=42,
            num_experiments=10,
            epochs_per_experiemnt=20,
            experiment_name="experiment",
            num_classes=10,
        )
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiments on {device}")
    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    print(
        f"Running {num_experiments} experiments for {experiment_name}, each with {epochs_per_experiemnt} epochs"
    )
    for i in range(num_experiments):
        print(f"Running experiment {i+1}/{num_experiments}")
        # Initialize the model
        model.apply(weight_reset)

        optimizer = def_optimizer(model)
        scheduler = def_scheduler(optimizer)
        train_dataloader = reset_dataloader(dataset[0], batch_size, shuffle=True)
        test_dataloader = reset_dataloader(dataset[1], batch_size, shuffle=False)
        history = train_and_test_model(
            model,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            scheduler,
            epochs_per_experiemnt,
            scheduler_sign=scheduler_sign,
            device=device,
            save_checkpoint=checkpoint_save,
            save_checkpoint_interval=1,
            checkpoint_save_dir=join(checkpoint_save_dir, experiment_name, "experiment_" + str(i + 1)),  # type: ignore
            output_logs=False,
            task=task,
            
        )
        eval_result = eval_model(
            model,
            test_dataloader,
            criterion,
            device=device,
            stage="Test",
            plot=False,
            task=task,
        )
        test_accuracy = eval_result["accuracy"]
        test_auc = eval_result["auc_score"]
        test_precision = eval_result["precision"]
        test_recall = eval_result["recall"]
        test_f1 = eval_result["f1"]

        acc_list.append(test_accuracy)
        auc_list.append(test_auc)
        precision_list.append(test_precision)
        recall_list.append(test_recall)
        f1_list.append(test_f1)

        print(
            f"Result for experiment {i+1}/{num_experiments}: acc: {test_accuracy :.5f} \tauc: {test_auc :.5f}"
        )
        print(
            f"recall: {test_recall :.5f} \tprecision: {test_precision :.5f} \tf1: {test_f1 :.5f}"
        )

    accuray = np.array(acc_list)
    acc_mean = np.mean(accuray)
    acc_var = np.var(accuray)
    acc_std = np.std(accuray)

    auc_score = np.array(auc_list)
    mean_auc = np.mean(auc_score)
    var_auc = np.var(auc_score)
    std_auc = np.std(auc_score)

    precision_score = np.array(precision_list)
    mean_precision = np.mean(precision_score)
    var_precision = np.var(precision_score)
    std_precision = np.std(precision_score)

    recall_score = np.array(recall_list)
    mean_recall = np.mean(recall_score)
    var_recall = np.var(recall_score)
    std_recall = np.std(recall_score)

    f1_score = np.array(f1_list)
    mean_f1 = np.mean(f1_score)
    var_f1 = np.var(f1_score)
    std_f1 = np.std(f1_score)

    print(
        f"result {num_experiments} experiments for {experiment_name}, each with {epochs_per_experiemnt} epochs: Accuracy {acc_mean:.5f} +/- {acc_std:.5f}"
    )
    acc_dict = {
        "mean": acc_mean,
        "variance": acc_var,
        "std": acc_std,
        "results": acc_list,
    }
    auc_dict = {
        "mean": mean_auc,
        "variance": var_auc,
        "std": std_auc,
        "results": auc_list,
    }

    precision_dict = {
        "mean": mean_precision,
        "variance": var_precision,
        "std": std_precision,
        "results": precision_list,
    }

    recall_dict = {
        "mean": mean_recall,
        "variance": var_recall,
        "std": std_recall,
        "results": recall_list,
    }

    f1_dict = {
        "mean": mean_f1,
        "variance": var_f1,
        "std": std_f1,
        "results": f1_list,
    }

    return {
        "accuracy": acc_dict,
        "auc_score": auc_dict,
        "precision": precision_dict,
        "recall": recall_dict,
        "f1": f1_dict,
    }
