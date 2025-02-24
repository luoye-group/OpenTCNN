import torch
from torch.optim.lr_scheduler import MultiStepLR


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizers_dict_medmnist(model_dict):
    """
    Get a dictionary of optimizers for the models.
    """
    return {
        model_name: torch.optim.Adam(model.parameters(), lr=0.001)
        for model_name, model in model_dict.items()
    }


def get_optimizers_medmnist(model):
    """
    Get an optimizer for the model.
    """
    return torch.optim.Adam(model.parameters(), lr=0.001)


def get_schedulers_dict_medmnist(optimizers_dict):
    """
    Get a dictionary of schedulers for the optimizers.
    """
    return {
        model_name: MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        for model_name, optimizer in optimizers_dict.items()
    }


def get_schedulers_medmnist(optimizer):
    """
    Get a scheduler for the optimizer.
    """
    return MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
