import os
import pickle


import pandas as pd


def show_repeat_result(res):
    """
    Display the result dictionary containing accuracy and AUC scores.

    Args:
        res (dict): The result dictionary containing accuracy and AUC scores.
        save_path (str, optional): The file path to save the result as a CSV file. Defaults to None.
    """
    print(
        f"{'Key':<10} {'Accuracy Mean':<15} {'Accuracy Std':<15} {'AUC Mean':<15} {'AUC Std':<15} {'F1 Mean':<15} {'F1 Std':<15}"
    )
    for key in res.keys():
        print(
            f"{key:<10} {res[key]['accuracy']['mean']:<15.5f} {res[key]['accuracy']['std']:<15.5f} {res[key]['auc_score']['mean']:<15.5f} {res[key]['auc_score']['std']:<15.5f} {res[key]['f1']['mean']:<15.5f} {res[key]['f1']['std']:<15.5f}"
        )


def save_result(res, save_name="log", save_dir="log"):
    """
    Save the result dictionary containing accuracy and AUC scores as a CSV file.

    Args:
        res (dict): The result dictionary containing accuracy and AUC scores.
        save_path (str, optional): The file path to save the result as a CSV file. Defaults to None.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    if save_name is not None:
        df = pd.DataFrame(res)
        dataframe_path = save_path + ".csv"
        df.to_csv(dataframe_path, index=False)
        print(f"Result saved to {dataframe_path}")
        # Save the result dictionary as a pickle file
        pickle_path = save_path + ".pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(res, f)
            print(f"Result saved as pickle file: {pickle_path}")
    else:
        print("No file path provided. Result not saved.")


def load_result(load_path):
    """
    Load the result dictionary containing accuracy and AUC scores from a pickle file.

    Args:
        load_path (str): The file path to load the result from.

    Returns:
        dict: The result dictionary containing accuracy and AUC scores.
    """
    with open(load_path, "rb") as f:
        res = pickle.load(f)
    return res


def print_consume_and_metrics(
    parameters_list,
    metric_list,
    model_name_list,
    xlabel="Number of Parameters",
    ylabel="Accuracy",
    figsize=(10, 4),
):
    """
    Print the parameters and accuracy of the models.

    Args:
        parameters_list (list): A list of the number of parameters in the models.
        accuracy_list (list): A list of the accuracy of the models.

    Returns:
    """
    print(f"{'Model':<20} {xlabel:<20} {ylabel:<20}")
    for params, metric, model_name in zip(
        parameters_list, metric_list, model_name_list
    ):
        print(f"{model_name:<20} {params:<20} {metric:<0}")


def show_test_result(result):
    """
    Display the test result dictionary containing accuracy and AUC scores.

    Args:
        result (dict): The test result dictionary containing accuracy and AUC scores.
    """
    from tcnn.utils.experiment.plot import plot_consume_and_metrics

    model_name_list = []
    parameters_list = []
    # macs_list = []
    accuracy_list = []
    print(
        f"{'Key':<10} {'Accuracy':<15} {'AUC':<15} {'Precision':<15} {'Recall':<15} {'F1':<15}"
    )
    for key in result.keys():
        print(
            f"{key:<10} {result[key]['performance']['accuracy']:<15.5f} {result[key]['performance']['auc_score']:<15.5f} {result[key]['performance']['precision']:<15.5f} {result[key]['performance']['recall']:<15.5f} {result[key]['performance']['f1']:<15.5f}"
        )
        model_name_list.append(key)
        # macs_list.append(result[key]["macs"])
        parameters_list.append(result[key]["params"])
        accuracy_list.append(result[key]["performance"]["accuracy"])

    plot_consume_and_metrics(
        parameters_list,
        accuracy_list,
        model_name_list,
        xlabel="Number of Parameters",
        ylabel="Accuracy(%)",
    )
    print_consume_and_metrics(
        parameters_list,
        accuracy_list,
        model_name_list,
        xlabel="Number of Parameters",
        ylabel="Accuracy(%)",
    )
    """
    plot_consume_and_metrics(
        macs_list, accuracy_list, model_name_list, xlabel="MACs", ylabel="Accuracy(%)"
    )
    print_consume_and_metrics(
        macs_list, accuracy_list, model_name_list, xlabel="MACs", ylabel="Accuracy(%)"
    )
    plot_consume_and_metrics(
        macs_list,
        parameters_list,
        model_name_list,
        xlabel="MACs",
        ylabel="Number of Parameters",
    )
    print_consume_and_metrics(
        macs_list,
        parameters_list,
        model_name_list,
        xlabel="MACs",
        ylabel="Number of Parameters",
    )
    """


def show_compare_result(source_result, quantized_result):
    print(
        f"{'Key':<10} {'Source Accuracy':<20} {'Quantized Accuracy':<20} {'Accuracy Change':<20} {'Source AUC':<20} {'Quantized AUC':<20} {'AUC Change':<20} {'Source Precision':<20} {'Quantized Precision':<20} {'Precision Change':<20} {'Source Recall':<20} {'Quantized Recall':<20} {'Recall Change':<20} {'Source F1':<20} {'Quantized F1':<20} {'F1 Change':<20}"
    )
    for key in source_result.keys():
        print(
            f"{key:<10} {source_result[key]['performance']['accuracy']:<20.5f} {quantized_result[key]['performance']['accuracy']:<20.5f} {quantized_result[key]['performance']['accuracy'] - source_result[key]['performance']['accuracy']:<20.5f} {source_result[key]['performance']['auc_score']:<20.5f} {quantized_result[key]['performance']['auc_score']:<20.5f} {quantized_result[key]['performance']['auc_score'] - source_result[key]['performance']['auc_score']:<20.5f} {source_result[key]['performance']['precision']:<20.5f} {quantized_result[key]['performance']['precision']:<20.5f} {quantized_result[key]['performance']['precision'] - source_result[key]['performance']['precision']:<20.5f} {source_result[key]['performance']['recall']:<20.5f} {quantized_result[key]['performance']['recall']:<20.5f} {quantized_result[key]['performance']['recall'] - source_result[key]['performance']['recall']:<20.5f} {source_result[key]['performance']['f1']:<20.5f} {quantized_result[key]['performance']['f1']:<20.5f} {quantized_result[key]['performance']['f1'] - source_result[key]['performance']['f1']:<20.5f}"
        )
