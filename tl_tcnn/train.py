from torchvision.transforms import v2 # Requires torchvision >= 0.16
import glob
import os
import time
from math import e
from unittest import result
from collections import defaultdict

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
from torchvision.transforms import v2
from tcnn.utils.experiment.train import find_latest_checkpoint, number_of_correct, get_likely_index
import pickle
from torch.utils.data import DataLoader, random_split, Subset

def soft_target_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy for soft targets (probability distributions).

    This is useful for MixUp/CutMix implementations that produce one-hot/soft labels.
    """
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def apply_mixup(data, target, alpha=1.0):
    """MixUp that keeps integer class targets.

    Returns (mixed_data, target_a, target_b, lam).
    """
    # Treat alpha<=0 as disabled (no-op)
    if not alpha or alpha <= 0:
        return data, target, target, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = data.size(0)
    index = torch.randperm(batch_size).to(data.device)
    mixed_data = lam * data + (1 - lam) * data[index, :]

    target_a = target
    target_b = target[index]
    return mixed_data, target_a, target_b, float(lam)


def apply_cutmix(data, target, alpha=1.0):
    """CutMix that keeps integer class targets.

    Returns (cutmixed_data, target_a, target_b, lam).
    """
    # Treat alpha<=0 as disabled (no-op). Important: CutMix would otherwise
    # still paste a random patch even when lam=1.0.
    if not alpha or alpha <= 0:
        return data, target, target, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = data.size(0)
    index = torch.randperm(batch_size).to(data.device)

    W, H = data.size(2), data.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = int(np.clip(cx - cut_w // 2, 0, W))
    bby1 = int(np.clip(cy - cut_h // 2, 0, H))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, W))
    bby2 = int(np.clip(cy + cut_h // 2, 0, H))

    # Apply patch and adjust lambda based on actual area
    data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    target_a = target
    target_b = target[index]
    return data, target_a, target_b, float(lam)

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    task="multiclass",
    num_classes=200, # Added for Mixup/CutMix
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
    grad_clip_max_norm=0.0, # Added for Gradient Clipping
    show_progress: bool = False,
    progress_desc: str = "train",
    log_interval: int = 50,
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
    
    # Initialize Mixup and CutMix
    # mixup = v2.MixUp(num_classes=num_classes, alpha=mixup_alpha)
    # cutmix = v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha)
    # cutmix_or_mixup = v2.RandomChoice([mixup, cutmix])
    
    correct = 0.0

    data_iter = enumerate(train_loader)
    if show_progress:
        data_iter = tqdm(
            data_iter,
            total=len(train_loader),
            desc=progress_desc,
            leave=False,
            mininterval=0.5,
        )

    for batch_idx, (data, target) in data_iter:
        data = data.to(device)
        target = target.to(device)

        # Randomly choose between Mixup, CutMix, or no augmentation.
        # Disable an augmentation by setting its alpha <= 0.
        mix_target_a = None
        mix_target_b = None
        mix_lam = None
        if task == "multiclass":
            mixup_on = bool(mixup_alpha) and mixup_alpha > 0
            cutmix_on = bool(cutmix_alpha) and cutmix_alpha > 0

            r = np.random.rand()
            if mixup_on and cutmix_on:
                # original behavior: 1/3 MixUp, 1/3 CutMix, 1/3 none
                if r < 0.33:
                    data, mix_target_a, mix_target_b, mix_lam = apply_mixup(data, target, alpha=mixup_alpha)
                elif r < 0.66:
                    data, mix_target_a, mix_target_b, mix_lam = apply_cutmix(data, target, alpha=cutmix_alpha)
            elif mixup_on and (not cutmix_on):
                # only MixUp enabled: 1/2 MixUp, 1/2 none
                if r < 0.5:
                    data, mix_target_a, mix_target_b, mix_lam = apply_mixup(data, target, alpha=mixup_alpha)
            elif cutmix_on and (not mixup_on):
                # only CutMix enabled: 1/2 CutMix, 1/2 none
                if r < 0.5:
                    data, mix_target_a, mix_target_b, mix_lam = apply_cutmix(data, target, alpha=cutmix_alpha)

        output = model(data)
        if isinstance(output, torch.Tensor) and output.ndim == 3 and output.shape[1] == 1:
            output = output.squeeze(1)

        # Loss: standard CE for normal labels; weighted CE for Mixup/CutMix
        if mix_target_a is None:
            loss = criterion(output, target)
        else:
            loss = mix_lam * criterion(output, mix_target_a) + (1.0 - mix_lam) * criterion(
                output, mix_target_b
            )

        if task == "multiclass":
            pred = get_likely_index(output)
        elif task == "binary":
            pred = torch.round(output)

        # Accuracy: weighted for Mixup/CutMix, normal otherwise
        if mix_target_a is None:
            correct += number_of_correct(pred, target)
        else:
            correct += mix_lam * number_of_correct(pred, mix_target_a)
            correct += (1.0 - mix_lam) * number_of_correct(pred, mix_target_b)

        optimizer.zero_grad()
        loss.backward()
        
        # --- Gradient Norm Clipping ---
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            
        optimizer.step()
        total_loss += loss.item()

        if show_progress and (log_interval and (batch_idx % log_interval == 0)):
            # Best-effort postfix, avoids expensive ops
            seen = (batch_idx + 1) * int(getattr(train_loader, "batch_size", 1) or 1)
            try:
                avg_loss = total_loss / max(1, (batch_idx + 1))
            except Exception:
                avg_loss = float("nan")
            try:
                acc = 100.0 * float(correct) / max(1.0, float(seen))
            except Exception:
                acc = float("nan")
            try:
                data_iter.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}")
            except Exception:
                pass
    # Note: Training accuracy is omitted here because Mixup labels are probabilistic. 
    # Use validation accuracy to judge performance.
    accuracy = 100.0 * correct / len(train_loader.dataset)

    return total_loss, accuracy



def train_one_epoch_depr(
    model,
    train_loader,
    criterion,
    optimizer,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    task="multiclass",
    num_classes=200, # Added for Mixup/CutMix
    mixup_alpha=1.0,
    cutmix_alpha=1.0,
    grad_clip_max_norm=1.0, # Added for Gradient Clipping
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
    # Infer number of output classes from model (more reliable than a hard-coded default)
    inferred_num_classes = None
    try:
        peek_batch = next(iter(train_loader))
        if isinstance(peek_batch, (list, tuple)) and len(peek_batch) >= 1:
            peek_data = peek_batch[0].to(device)
            with torch.no_grad():
                was_training = model.training
                model.eval()
                peek_out = model(peek_data)
                if was_training:
                    model.train()
            if isinstance(peek_out, torch.Tensor) and peek_out.ndim == 3 and peek_out.shape[1] == 1:
                peek_out = peek_out.squeeze(1)
            if isinstance(peek_out, torch.Tensor) and peek_out.ndim >= 2:
                inferred_num_classes = int(peek_out.shape[-1])
    except StopIteration:
        inferred_num_classes = None

    # Use inferred class count when available (prevents MixUp/CutMix target dim mismatch)
    num_classes_aug = inferred_num_classes or num_classes

    model.train()
    total_loss = 0
    
    # Initialize Mixup and CutMix (only if enabled)
    enabled_augs = []
    if mixup_alpha and mixup_alpha > 0:
        enabled_augs.append(v2.MixUp(num_classes=num_classes_aug, alpha=mixup_alpha))
    if cutmix_alpha and cutmix_alpha > 0:
        enabled_augs.append(v2.CutMix(num_classes=num_classes_aug, alpha=cutmix_alpha))
    cutmix_or_mixup = v2.RandomChoice(enabled_augs) if enabled_augs else None
    
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        if cutmix_or_mixup is not None and task == "multiclass":
            data, target = cutmix_or_mixup(data, target)
        
        # # Randomly choose between Mixup, CutMix, or no augmentation (1/3 chance each)
        # r = np.random.rand()
        # if r < 0.33:
        #     data, target = apply_mixup(data, target, alpha=mixup_alpha, num_classes=num_classes)
        # elif r < 0.66:
        #     data, target = apply_cutmix(data, target, alpha=cutmix_alpha, num_classes=num_classes)
        # else:
        #     # If no Mixup/CutMix, convert to one-hot anyway so criterion works consistently
        #     target = torch.nn.functional.one_hot(target, num_classes).float()

        # Keep a hard-label view for accuracy even when using soft labels (Mixup/CutMix)
        # - If target is one-hot / soft distribution: argmax -> class index
        # - If target is already class indices: keep as-is
        hard_target = target
        if isinstance(target, torch.Tensor) and target.ndim > 1:
            hard_target = target.argmax(dim=1)

        output = model(data)
        if isinstance(output, torch.Tensor) and output.ndim == 3 and output.shape[1] == 1:
            output = output.squeeze(1)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        if isinstance(target, torch.Tensor) and target.ndim > 1:
            # MixUp/CutMix returns soft targets; nn.CrossEntropyLoss expects class indices.
            loss = soft_target_cross_entropy(output, target)
        else:
            loss = criterion(output, target)
        if task == "multiclass":
            pred = get_likely_index(output)
        elif task == "binary":
            pred = torch.round(output)
        correct += number_of_correct(pred, hard_target)
        optimizer.zero_grad()
        loss.backward()
        
        # --- Gradient Norm Clipping ---
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            
        optimizer.step()
        total_loss += loss.item()
    # Note: Training accuracy is omitted here because Mixup labels are probabilistic. 
    # Use validation accuracy to judge performance.
    accuracy = 100.0 * correct / len(train_loader.dataset)

    return total_loss, accuracy


def test_one_epoch(
    model,
    epoch,
    dataloader,
    crtiterion,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    task="multiclass",
    show_progress: bool = False,
    progress_desc: str = "val",
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
    correct_top1 = 0
    correct_top5 = 0
    total_samples = len(dataloader.dataset)
    
    total_loss = 0

    data_iter = dataloader
    if show_progress:
        data_iter = tqdm(
            data_iter,
            total=len(dataloader),
            desc=progress_desc,
            leave=False,
            mininterval=0.5,
        )

    for data, target in data_iter:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        if isinstance(output, torch.Tensor) and output.ndim == 3 and output.shape[1] == 1:
            output = output.squeeze(1)
        loss = crtiterion(output, target)

        if task == "multiclass":
            # pred = get_likely_index(output)
            # Top-1 Accuracy
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct_top1 += pred.eq(target.view(1, -1).expand_as(pred)).sum().item()

            # Top-5 Accuracy
            _, pred5 = output.topk(5, 1, True, True)
            pred5 = pred5.t()
            correct_top5 += pred5.eq(target.view(1, -1).expand_as(pred5)).sum().item()
        elif task == "binary":
            pred = torch.round(output)
            # pred = torch.round(torch.sigmoid(output))
            correct_top1 += (pred == target).sum().item()
            correct_top5 = correct_top1 # Top-5 doesn't apply to binary
        # correct += number_of_correct(pred, target)

        total_loss += loss.item()

        if show_progress:
            try:
                # Show running top1/top5
                seen = max(1, int(total_samples))
                acc1_run = 100.0 * float(correct_top1) / seen
                acc5_run = 100.0 * float(correct_top5) / seen
                data_iter.set_postfix(acc1=f"{acc1_run:.2f}", acc5=f"{acc5_run:.2f}")
            except Exception:
                pass

    # accuracy = 100.0 * correct / len(dataloader.dataset)
    acc1 = 100.0 * correct_top1 / total_samples
    acc5 = 100.0 * correct_top5 / total_samples
    # return accuracy, total_loss
    return acc1, acc5, total_loss

def train_and_test_model(
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    num_classes=200, # Added for Mixup/CutMix
    scheduler_sign=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    output_logs=True,
    save_checkpoint=False,
    save_checkpoint_interval=10,
    checkpoint_save_dir="./checkpoints/",
    task="multiclass",
    mixup_alpha=1.0,
    cutmix_alpha=1.0,
    grad_clip_max_norm=1.0, # Added for Gradient Clipping
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
        # "test_acc1": [],
        "test_acc5": [],
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
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            grad_clip_max_norm=grad_clip_max_norm,
        )

        test_accuracy, test_acc5, test_loss = test_one_epoch(
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
                f"Epoch: {epoch} Train accuracy: {train_accuracy:.5f}%  Test accuracy: {test_accuracy:.5f}%  Test Acc@5: {test_acc5:.2f}%  Best accuracy: {best_accuracy:.5f}%"
            )

        # append train loss , test loss, train accuracy, test accuracy to history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)
        history["test_acc5"].append(test_acc5)
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


def _extract_labels(dataset):
    """Extract classification labels from a dataset for stratified splitting."""
    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(x) for x in dataset.labels]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def _build_split_indices_random(total_size, split_target, train_ratio, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_size).tolist()

    if split_target == "train_val":
        train_size = int((1.0 - val_ratio) * total_size)
        val_size = total_size - train_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError("Invalid split sizes for train_val. Adjust val_ratio.")
        return {
            "train_indices": indices[:train_size],
            "val_indices": indices[train_size:],
        }

    if split_target != "train_val_test":
        raise ValueError("split_target must be 'train_val' or 'train_val_test'.")

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            "Train/val sizes must be > 0. Adjust train_ratio/val_ratio."
        )
    if test_size <= 0 and split_target == "train_val_test":
        raise ValueError(
        "test sizes must be > 0. Adjust train_ratio/val_ratio."
        )
    return {
        "train_indices": indices[:train_size],
        "val_indices": indices[train_size:train_size + val_size],
        "test_indices": indices[train_size + val_size:],
    }


def _build_split_indices_stratified(dataset, split_target, train_ratio, val_ratio, seed):
    labels = _extract_labels(dataset)
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[int(label)].append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for _, indices in sorted(class_to_indices.items()):
        n = len(indices)
        perm = torch.randperm(n, generator=generator).tolist()

        if split_target == "train_val":
            if n < 2:
                raise ValueError("Each class needs at least 2 samples for train_val split.")
            n_val = min(max(1, int(n * val_ratio)), n - 1)
            n_train = n - n_val

            val_indices.extend(indices[i] for i in perm[:n_val])
            train_indices.extend(indices[i] for i in perm[n_val:n_val + n_train])
            continue

        if split_target != "train_val_test":
            raise ValueError("split_target must be 'train_val' or 'train_val_test'.")

        if n < 3:
            raise ValueError(
                "Each class needs at least 3 samples for stratified train_val_test split."
            )

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(1, min(n_val, n - 2))
            n_train = max(1, n - n_val - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            raise ValueError("Invalid class-wise split produced empty test subset.")

        val_indices.extend(indices[i] for i in perm[:n_val])
        train_indices.extend(indices[i] for i in perm[n_val:n_val + n_train])
        test_indices.extend(indices[i] for i in perm[n_val + n_train:])

    out = {
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    if split_target == "train_val_test":
        out["test_indices"] = test_indices
    return out


def create_and_save_split(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    save_path='./checkpoints/split_indices.pkl',
    seed: int = 42,
    strategy: str = "random",
    split_target: str = "train_val_test",
):
    """Create and persist split indices (PKL) for train/val or train/val/test.

    Args:
        dataset: Dataset to split.
        train_ratio: Used when split_target == "train_val_test".
        val_ratio: Validation ratio.
        save_path: PKL path to save indices.
        seed: Random seed.
        strategy: "random" or "stratified".
        split_target: "train_val" or "train_val_test".
    """
    total_size = len(dataset)
    if total_size <= 1:
        raise ValueError("Dataset must contain at least 2 samples.")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in (0, 1).")
    if split_target == "train_val_test" and (train_ratio <= 0 or train_ratio >= 1):
        raise ValueError("train_ratio must be in (0, 1) for train_val_test.")
    if split_target == "train_val_test" and (train_ratio + val_ratio >= 1):
        raise ValueError("train_ratio + val_ratio must be < 1 for train_val_test.")

    strategy = strategy.strip().lower()
    if strategy == "stratified":
        indices = _build_split_indices_stratified(
            dataset=dataset,
            split_target=split_target,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    elif strategy == "random":
        indices = _build_split_indices_random(
            total_size=total_size,
            split_target=split_target,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    else:
        raise ValueError("strategy must be 'random' or 'stratified'.")

    split_info = {
        "split_target": split_target,
        "strategy": strategy,
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "total_size": int(total_size),
        **indices,
    }
    split_info["train_size"] = len(split_info["train_indices"])
    split_info["val_size"] = len(split_info["val_indices"])
    if "test_indices" in split_info:
        split_info["test_size"] = len(split_info["test_indices"])

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(split_info, f)

    return split_info


def load_split(dataset, split_path='./checkpoints/split_indices.pkl'):
    """Load a split PKL and materialize dataset subsets.

    Returns:
        (train_dataset, val_dataset, test_dataset, split_info)
        where `test_dataset` can be None when split_target is train_val.
    """
    with open(split_path, 'rb') as f:
        split_info = pickle.load(f)

    train_dataset = Subset(dataset, [int(i) for i in split_info['train_indices']])
    val_dataset = Subset(dataset, [int(i) for i in split_info['val_indices']])

    test_dataset = None
    if 'test_indices' in split_info:
        test_dataset = Subset(dataset, [int(i) for i in split_info['test_indices']])

    return train_dataset, val_dataset, test_dataset, split_info


def load_or_create_split(
    dataset,
    split_path: str = './checkpoints/split_indices.pkl',
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    strategy: str = "stratified",
    split_target: str = "train_val",
):
    """Load a split PKL if exists, else create one, then return subsets."""
    if os.path.exists(split_path):
        train_dataset, val_dataset, test_dataset, split_info = load_split(
            dataset=dataset,
            split_path=split_path,
        )
    else:
        split_info = create_and_save_split(
            dataset=dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            save_path=split_path,
            seed=seed,
            strategy=strategy,
            split_target=split_target,
        )
        train_dataset, val_dataset, test_dataset, split_info = load_split(
            dataset=dataset,
            split_path=split_path,
        )

    return train_dataset, val_dataset, test_dataset, split_info


def prepare_datasets_by_split_mode(
    split_mode: str,
    *,
    split_path: str = './checkpoints/split_indices.pkl',
    strategy: str = "stratified",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    unsplit_dataset=None,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
):
    """Prepare datasets according to user-selected split mode.

    Modes:
      - "unsplit": one dataset only -> split into train/val/test.
      - "train_test": have train+test -> split train into train/val.
      - "train_val_test": already split -> pass through.
    """
    mode = split_mode.strip().lower()

    if mode == "unsplit":
        if unsplit_dataset is None:
            raise ValueError("unsplit mode requires `unsplit_dataset`.")
        train_ds, val_ds, test_ds, split_info = load_or_create_split(
            dataset=unsplit_dataset,
            split_path=split_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            strategy=strategy,
            split_target="train_val_test",
        )
        return train_ds, val_ds, test_ds, split_info

    if mode == "train_test":
        if train_dataset is None or test_dataset is None:
            raise ValueError("train_test mode requires `train_dataset` and `test_dataset`.")
        train_ds, val_ds, _, split_info = load_or_create_split(
            dataset=train_dataset,
            split_path=split_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            strategy=strategy,
            split_target="train_val",
        )
        return train_ds, val_ds, test_dataset, split_info

    if mode == "train_val_test":
        if train_dataset is None or val_dataset is None or test_dataset is None:
            raise ValueError(
                "train_val_test mode requires `train_dataset`, `val_dataset`, and `test_dataset`."
            )
        split_info = {
            "split_target": "pre_split",
            "strategy": "none",
            "seed": None,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        }
        return train_dataset, val_dataset, test_dataset, split_info

    raise ValueError("split_mode must be one of: unsplit, train_test, train_val_test.")


# Backward-compatible wrapper
def create_and_save_stratified_train_val_split(
    dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
    save_path: str = './checkpoints/split_indices.pkl',
):
    return create_and_save_split(
        dataset=dataset,
        val_ratio=val_ratio,
        save_path=save_path,
        seed=seed,
        strategy="stratified",
        split_target="train_val",
    )


# Backward-compatible wrapper
def load_or_create_stratified_train_val_split(
    dataset,
    split_path: str = './checkpoints/split_indices.pkl',
    val_ratio: float = 0.2,
    seed: int = 42,
):
    train_dataset, val_dataset, _, split_info = load_or_create_split(
        dataset=dataset,
        split_path=split_path,
        val_ratio=val_ratio,
        seed=seed,
        strategy="stratified",
        split_target="train_val",
    )
    return train_dataset, val_dataset, split_info
