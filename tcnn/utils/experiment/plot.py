from itertools import cycle
from turtle import color

from matplotlib import markers
from matplotlib import pyplot as plt
from pyexpat import model


def visual_history(
    historys_list, legend_list, key, ylabel, xlabel="Epoch", figsize=(18, 8)
):
    """
    Visualize the history of multiple experiments.

    Args:
        historys_list (list): A list of history objects containing training and testing metrics.
        legend_list (list): A list of strings representing the legend for each experiment.
        key (str): The key to visualize.
        ylabel (str): The label for the y-axis.
        xlabel (str): The label for the x-axis.
        figsize (tuple): The size of the figure.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    index = 0
    markers = cycle(
        ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]
    )
    for history in historys_list:
        plt.plot(
            history[key], label=legend_list[index] + " " + key, marker=next(markers)
        )
        index += 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{key}: {ylabel}~{xlabel}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_history_list(historys_list, legend_list):
    """
    Plot the history of multiple experiments.

    Args:
        historys_list (list): A list of history objects containing training and testing metrics.
        legend_list (list): A list of strings representing the legend for each experiment.

    Raises:
        AssertionError: If the length of `historys_list` is not equal to the length of `legend_list`.

    Returns:
        None
    """
    assert len(historys_list) == len(legend_list)
    visual_history(
        historys_list, legend_list, key="train_loss", ylabel="Loss", xlabel="Epoch"
    )
    visual_history(
        historys_list, legend_list, key="test_loss", ylabel="Loss", xlabel="Epoch"
    )
    visual_history(
        historys_list,
        legend_list,
        key="train_accuracy",
        ylabel="Accuracy",
        xlabel="Epoch",
    )
    visual_history(
        historys_list,
        legend_list,
        key="test_accuracy",
        ylabel="Accuracy",
        xlabel="Epoch",
    )
    visual_history(
        historys_list,
        legend_list,
        key="lr",
        ylabel="Learning Rate",
        xlabel="Epoch",
    )


def plot_history_dict(history_dict):
    """
    Plot the history of multiple experiments.

    Args:
        history_dict (dict): A dictionary containing the history objects.

    Returns:
        None
    """
    legend_list = list(history_dict.keys())
    historys_list = list(history_dict.values())
    plot_history_list(historys_list, legend_list)


def plot_history(history, key, figsize=(18, 8)):
    """
    Plots the training and testing history.

    Args:
        history (dict): A dictionary containing the training and testing loss and accuracy.

    Returns:
        None

    Raises:
        None

    Examples:
        # Plot the training and testing history
        plot_history(history)
    """
    plt.figure(figsize=figsize)
    plt.suptitle(f"Training and Testing History for {key}")
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["test_loss"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training and Testing Loss for {key}")
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="train")
    plt.plot(history["test_accuracy"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Training and Testing Accuracy for {key}")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=figsize)
    plt.plot(history["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"Learning Rate for {key}")
    plt.show()


def plot_experiment_errorbar(
    res,
    metric_key,
    baseline_key,
    ylabel,
    xlabel="Experiment",
    title="Experiment Result for ",
    figsize=(18, 8),
):
    """
    Plot the experiment results with error bars.

    Args:
        res (dict): A dictionary containing the experiment results.
        metric_key (str): The key to access the metric values in the experiment results.
        baseline_key (str): The key to access the baseline metric value in the experiment results.
        ylabel (str): The label for the y-axis.
        xlabel (str, optional): The label for the x-axis. Defaults to "Experiment".
        title (str, optional): The title of the plot. Defaults to "Experiment Result for ".

    Returns:
        None
    """

    mean_result = []
    std_dev = []
    for key in res.keys():
        mean_result.append(res[key][metric_key]["mean"])
        std_dev.append(res[key][metric_key]["std"])
    plt.figure(figsize=figsize)
    plt.errorbar(res.keys(), mean_result, yerr=std_dev, fmt="o", color="b")
    baseline = res[baseline_key][metric_key]["mean"]
    plt.axhline(y=baseline, color="r", linestyle="--")

    plt.title(title + metric_key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_consume_and_metrics(
    parameters_list,
    metric_list,
    model_name_list,
    xlabel="Number of Parameters",
    ylabel="Accuracy",
    figsize=(10, 4),
):
    """
    Plot the parameters and accuracy of the models.

    Args:
        parameters_list (list): A list of the number of parameters in the models.
        accuracy_list (list): A list of the accuracy of the models.

    Returns:
        None
    """
    assert len(parameters_list) == len(metric_list)
    assert len(parameters_list) == len(model_name_list)
    markers = cycle(
        ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]
    )
    colors = cycle(["b", "g", "r", "c", "m", "y", "k"])

    plt.figure(figsize=figsize)
    for i in range(len(parameters_list)):
        plt.scatter(
            parameters_list[i],
            metric_list[i],
            label=model_name_list[i],
            marker=next(markers),
            color=next(colors),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"{ylabel} vs {xlabel}")
    plt.show()


def plot_macs_vs_batch(
    batch_list, mac_dict, ylabel="Macs", xlabel="Batch Size", figsize=(12, 8)
):
    """
    Plot the MACs vs Batch Size.

    Args:
        batch_list (list): A list of batch sizes.
        mac_dict (dict): A dictionary containing the MACs for each batch size.

    Returns:
        None
    """
    markers = cycle(
        ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]
    )
    colors = cycle(["b", "g", "r", "c", "m", "y", "k"])
    plt.figure(figsize=figsize)
    for key in mac_dict.keys():
        assert len(batch_list) == len(mac_dict[key])
        plt.plot(
            batch_list,
            mac_dict[key],
            label=key,
            marker=next(markers),
            color=next(colors),
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.legend()
    plt.show()
