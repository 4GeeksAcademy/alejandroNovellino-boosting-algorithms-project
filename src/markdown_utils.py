"""
Utility functions for Markdown elements.
"""

import numpy as np
import numpy.typing as npt
from IPython.core.display_functions import display
from IPython.display import Markdown


def show_comparison_table(metric_names: list[str], default_metrics: list[float], optimized_metrics: list[float]) -> None:
    """
    Creates and show a Markdown table comparing default and optimized model metrics.

    Args:
        metric_names (list[str]): List of metric names.
        default_metrics (list[float]): List of metric values for the default model.
        optimized_metrics (list[float]): List of metric values for the optimized model.

    Returns:
        str: A Markdown table as a string.

    Raise:
        Exception: the metrics length is different from the default or optimized metrics.
    """

    if len(metric_names) != len(default_metrics) or len(metric_names) != len(optimized_metrics):
        raise Exception("Error: Metric lists must have the same length.")

    markdown_table = "| Metric | Default Model | Optimized Model |\n"
    markdown_table += "|---|---|---|\n"

    for i in range(len(metric_names)):
        markdown_table += f"| {metric_names[i]} | {np.round(default_metrics[i], 2)} | {np.round(optimized_metrics[i], 2)} |\n"

    # display the table
    display(Markdown(markdown_table))

    return None


def show_confusion_matrix_analysis(confusion_matrix: npt.NDArray[np.float64], positive_label_meaning: str, negative_label_meaning: str) -> None:
    """
    Creates and show a Markdown table comparing default and optimized model metrics.

    Args:
        confusion_matrix (npt.NDArray[np.float64]): Confusion matrix to show analysis, this matrix comes from the
            sklearn metrics functions.
        positive_label_meaning (str): Meaning of the positive (1) label in the confusion matrix.
        negative_label_meaning (str): Meaning of the negative (0) label in the confusion matrix.

    Returns:
        str: A Markdown table as a string.

    Raise:
        Exception: the metrics length is different from the default or optimized metrics.
    """

    try:
        analysis = f'''
        The interpretation of a confusion matrix is as follows:

            - **True positive (TP)**: corresponds to the number **({confusion_matrix[1][1]})** and are the cases where the model predicted positive **({positive_label_meaning})** and the actual class is also positive.
            - **True negative (TN)**: Corresponds to the number **({confusion_matrix[0][0]})** and are the cases where the model predicted negative **({negative_label_meaning})** and the actual class is also negative.
            - **False positive (FP)**: Corresponds to the number **({confusion_matrix[0][1]})** and are the cases in which the model predicted positive, but the actual class is negative.
            - **False negative (FN)**: Corresponds to the number **({confusion_matrix[1][0]})** and are the cases where the model predicted negative, but the actual class is positive.
        '''

        # display the table
        display(Markdown(analysis))
    except:
        display(Markdown('Error: The confusion matrix is not a 2x2 numpy array.'))

    return None
