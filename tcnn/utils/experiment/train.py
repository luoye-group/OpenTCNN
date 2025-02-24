import glob
import os
import time
from math import e
from unittest import result

import cpuinfo
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from tcnn.utils.experiment.model import count_parameters


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    task="multiclass",
):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        epoch (int): The current epoch number.
        train_loader (torch.utils.data.DataLoader): The dataloader for training data.
        criterion (torch.nn.Module): The loss function for the model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler for the optimizer.
        log_interval (int, optional): The interval at which to log the training loss. Defaults to 100.
        device (torch.device, optional): The device to use for training. Defaults to "cuda" if available, otherwise "cpu".


    Returns:
        float: The total loss for the epoch.

    Raises:
        None

    Examples:
        # Train the model for one epoch
        loss = train(model, epoch, train_loader, optimizer, scheduler)
    """
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = criterion(output, target)
        if task == "multiclass":
            pred = get_likely_index(output)
        elif task == "binary":
            pred = torch.round(output)
        correct += number_of_correct(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    accuracy = 100.0 * correct / len(train_loader.dataset)

    return total_loss, accuracy


def number_of_correct(pred, target):
    """
    Counts the number of correct predictions.

    Args:
        pred (torch.Tensor): The predicted labels.
        target (torch.Tensor): The target labels.

    Returns:
        int: The number of correct predictions.

    Raises:
        None

    Examples:
        # Count the number of correct predictions
        correct = number_of_correct(pred, target)
    """
    return pred.eq(target).sum().item()


def get_likely_index(tensor):
    """
    Finds the most likely label index for each element in the batch.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The most likely label index for each element in the batch.

    Raises:
        None

    Examples:
        # Find the most likely label index
        index = get_likely_index(tensor)
    """
    return tensor.argmax(dim=-1)


def test_one_epoch(
    model,
    epoch,
    dataloader,
    crtiterion,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    task="multiclass",
):
    """
    Tests the model on the test data.

    Args:
        model (torch.nn.Module): The model to be tested.
        epoch (int): The current epoch number.
        dataloader (torch.utils.data.DataLoader): The dataloader for testing data.
        crtiterion (torch.nn.Module): The loss function for the model.
        device (torch.device, optional): The device to use for testing. Defaults to "cuda" if available, otherwise "cpu".

    Returns:
        tuple: A tuple containing the accuracy and total loss.

    Raises:
        None

    Examples:
        # Test the model on the test data
        accuracy, loss = test(model, epoch, test_dataloader)
    """
    model.eval()
    correct = 0
    total_loss = 0
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = crtiterion(output, target)

        if task == "multiclass":
            pred = get_likely_index(output)
        elif task == "binary":
            pred = torch.round(output)
        correct += number_of_correct(pred, target)

        total_loss += loss.item()

    accuracy = 100.0 * correct / len(dataloader.dataset)
    return accuracy, total_loss


def train_and_test_model(
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    scheduler_sign=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    output_logs=True,
    save_checkpoint=False,
    save_checkpoint_interval=10,
    checkpoint_save_dir="./checkpoints/",
    task="multiclass",
):
    """
    Trains and tests the given model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for training data.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for testing data.
        criterion (torch.nn.Module): The loss function for the model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler for the optimizer.
        epochs (int): The number of epochs to train the model.
        scheduler_sign (str, optional): The sign to be used for the scheduler. Defaults to None.
        device (torch.device, optional): The device to use for training and testing. Defaults to "cuda" if available, otherwise "cpu".
        output_logs (bool, optional): Whether to output logs. Defaults to True.
        save_checkpoint (bool, optional): Whether to save the model checkpoint. Defaults to False.
        checkpoint_path (str, optional): The path to save the model checkpoint. Defaults to None.
        save_checkpoint_interval (int, optional): The interval at which to save the model checkpoint. Defaults to 10.
        checkpoint_save_dir (str, optional): The directory to save the model checkpoint. Defaults to './checkpoints/'.

    Returns:
        dict: A dictionary containing the training and testing loss and accuracy.

    Raises:
        None

    Examples:
        # Create model, dataloaders, optimizer, and scheduler
        model = MyModel()
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Train and test the model
        history = train_and_test_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, epochs=100)
    """
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "test_loss": [],
        "lr": [],
    }
    if save_checkpoint:
        os.makedirs(checkpoint_save_dir, exist_ok=True)
        print(f"Checkpoints will be saved in {checkpoint_save_dir}")
        checkpoint_path = find_latest_checkpoint(checkpoint_save_dir)
        if checkpoint_path is not None:
            print(f"Found latest checkpoint at {checkpoint_path}")
        else:
            print("No checkpoints found")

    best_accuracy = 0
    if torch.cuda.device_count() >= 1:
        model = torch.nn.DataParallel(model).to(device)

    print(f"Training on {device} and {torch.cuda.device_count()} GPUs:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Device : {cpuinfo.get_cpu_info()['brand_raw']}")

    model_parameters = count_parameters(model)
    print(f"Model parameters: {model_parameters}({model_parameters/(1024 ** 2):.5f}MB)")

    start_time = time.time()
    if save_checkpoint and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epochs = checkpoint["epoch"] + 1
        history["train_loss"] = checkpoint["train_loss"]
        history["test_loss"] = checkpoint["test_loss"]
        history["train_accuracy"] = checkpoint["train_accuracy"]
        history["test_accuracy"] = checkpoint["test_accuracy"]
        # compatible with old checkpoint
        if "lr" in checkpoint.keys():
            history["lr"] = checkpoint["lr"]

        print(
            f"Loaded checkpoint from epoch {start_epochs} and continue training to {epochs}"
        )
    else:
        start_epochs = 0
        print(f"Start training from epoch 0 to {epochs}")

    for epoch in tqdm(range(start_epochs, epochs)):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_dataloader,
            criterion,
            optimizer,
            device=device,
            task=task,
        )

        test_accuracy, test_loss = test_one_epoch(
            model, epoch, test_dataloader, criterion, device=device, task=task
        )

        "scheduler"
        if scheduler is not None:
            if scheduler_sign == "val_acc":
                scheduler.step(test_accuracy)
            else:
                scheduler.step()

        "show best accuracy"
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if save_checkpoint:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": history["train_loss"],
                        "test_loss": history["test_loss"],
                        "train_accuracy": history["train_accuracy"],
                        "test_accuracy": history["test_accuracy"],
                    },
                    f"{checkpoint_save_dir}/best_checkpoint.pth",
                )
                print(
                    f"saved best checkpoint at epoch {epoch} to {checkpoint_save_dir}"
                )
        if output_logs:
            print(
                f"Epoch: {epoch} Train accuracy: {train_accuracy:.5f}%  Test accuracy: {test_accuracy:.5f}%  Best accuracy: {best_accuracy:.5f}%"
            )

        # append train loss , test loss, train accuracy, test accuracy to history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # Save the model checkpoint
        if save_checkpoint and epoch % save_checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": history["train_loss"],
                    "test_loss": history["test_loss"],
                    "train_accuracy": history["train_accuracy"],
                    "test_accuracy": history["test_accuracy"],
                    "lr": history["lr"],
                },
                f"{checkpoint_save_dir}/checkpoint_{epoch}.pth",
            )
            print(f"Saved checkpoint at epoch {epoch} to {checkpoint_save_dir}")
            if epoch - save_checkpoint_interval >= 0:
                os.remove(
                    f"{checkpoint_save_dir}/checkpoint_{epoch-save_checkpoint_interval}.pth"
                )
                print(
                    f"Removed checkpoint at epoch {epoch-save_checkpoint_interval} from {checkpoint_save_dir}"
                )

    duration_time = time.time() - start_time
    print(
        f"Training time: {duration_time:.2f}s and Average time per epoch: {duration_time / epochs:.2f}s"
    )

    return history


def eval_model(
    model,
    test_dataloader,
    criterion,
    task="multiclass",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    stage="Test",
    plot=True,
):
    """
    Evaluates the model on the test data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for testing data.
        device (torch.device, optional): The device to use for evaluation. Defaults to "cuda" if available, otherwise "cpu".
        stage (str, optional): The stage of evaluation. Defaults to "Test".
        plot (bool, optional): Whether to plot the confusion matrix. Defaults to True.

    Returns:
        float: The accuracy of the model on the test data.

    Examples:
        # Evaluate the model on the test data
        accuracy = eval_model(model, test_dataloader)
    """
    model.eval()
    model = torch.nn.DataParallel(model).to(device)
    correct = 0
    total_loss = 0
    all_preds = []
    all_preds_probs = []
    all_targets = []
    for data, target in test_dataloader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()

        if task == "multiclass":
            pred = get_likely_index(output)
        elif task == "binary":
            pred = torch.round(output)
        correct += number_of_correct(pred, target)

        if task == "multiclass":
            all_preds_probs.extend(F.softmax(output, dim=1).cpu().detach().numpy())
        elif task == "binary":
            all_preds_probs.extend(output.cpu().detach().numpy())

        all_preds.extend(pred.cpu().detach().numpy())
        all_targets.extend(target.cpu().detach().numpy())

    accuracy = 100.0 * correct / len(test_dataloader.dataset)
    print(
        f"{stage} Accuracy: {correct}/{len(test_dataloader.dataset)} ({accuracy:.5f}%)\n"
    )
    # Compute AUC score
    # Compute precision, recall, and F1 score
    if task == "multiclass":
        try:
            auc_score = roc_auc_score(all_targets, all_preds_probs, multi_class="ovr")
        except ValueError:
            auc_score = -1

        try:
            precision = precision_score(all_targets, all_preds, average="macro")
        except ValueError:
            precision = -1

        try:
            recall = recall_score(all_targets, all_preds, average="macro")
        except ValueError:
            recall = -1

        try:
            f1 = f1_score(all_targets, all_preds, average="macro")
        except ValueError:
            f1 = -1
    elif task == "binary":
        try:
            auc_score = roc_auc_score(all_targets, all_preds_probs)
        except ValueError:
            auc_score = -1

        try:
            precision = precision_score(all_targets, all_preds)
        except ValueError:
            precision = -1

        try:
            recall = recall_score(all_targets, all_preds)
        except ValueError:
            recall = -1

        try:
            f1 = f1_score(all_targets, all_preds)
        except ValueError:
            f1 = -1

    print(
        f"{stage} AUC Score: {auc_score:.5f} Precision: {precision:.5f} Recall: {recall:.5f} F1: {f1:.5f}"
    )
    if plot:
        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        print("Confusion Matrix: \n", cm)

        # Compute classification report
        cr = classification_report(all_targets, all_preds)
        print("Classification Report: \n", cr)
        # Visualize confusion matrix
        plt.imshow(cm, interpolation="nearest", cmap="hot_r")
        plt.title("Confusion matrix")
        plt.colorbar()
        if task == "multiclass":
            tick_marks = np.arange(len(set(all_targets)))
        elif task == "binary":
            tick_marks = np.arange(2)
        plt.xticks(tick_marks, rotation=45)
        plt.yticks(tick_marks)
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    result = {
        "accuracy": accuracy,
        "auc_score": auc_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return result


def find_latest_checkpoint(checkpoint_save_dir):
    # 使用 glob 来找到所有的 pth 文件
    checkpoints = glob.glob(os.path.join(checkpoint_save_dir, "checkpoint_*.pth"))

    if not checkpoints:
        # 如果没有找到任何 pth 文件，返回 None
        return None

    # 使用 max 函数和 os.path.getmtime 函数来找到最新的 pth 文件
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)

    return latest_checkpoint
