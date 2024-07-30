import os
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame

from utils.summary_writer import read_epoch_summary_files_to_df, read_update_step_summary_files_to_df


def plot_summary(summary_df: DataFrame, title: str, xlabel: str = "Epoch", ylabel: str = "Accuracy",
                 y_log_scale: bool = False, loss: bool = False):
    ax = plt.figure().gca()
    ax.grid(visible=True)
    min_y_value = round(summary_df.value.min(), 1)
    max_y_value = round(summary_df.value.max(), 1)

    if y_log_scale:
        ax.set_yscale('symlog')
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_yticks(np.append(ax.get_yticks(), [min_y_value * 1.1, 2, max_y_value]))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    n_epochs = max(list(summary_df.step))
    sns.set_theme(palette="colorblind", style="whitegrid")
    plot = sns.lineplot(x="step", y="value", hue="model", data=summary_df)
    plot.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plot.set_xlim(0, n_epochs)
    if y_log_scale:
        plt.gca().set_ylim(bottom=min_y_value * 1.1, top=max_y_value * 1.1)
    if loss:
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")


def plot_comparison(
        summary_paths: List[str],
        n_runs: int,
        n_epochs: int,
        title: Union[str, None] = None,
        model_names: Union[List[str], None] = None,
        xlabel: str = "Epoch",
        ylabel: str = "Accuracy",
        y_log_scale: bool = False,
        loss: bool = False
):
    if model_names is None:
        model_names = [os.path.split(path)[-1] for path in summary_paths]

    if title is None:
        title = "Model comparison"

    if len(summary_paths) != len(model_names):
        raise ValueError("summary_paths and model_names must have the same size")

    if loss:
        comparison_df = pd.concat(
            [read_update_step_summary_files_to_df(path, model_names[i], n_runs, n_epochs) for i, path in
             enumerate(summary_paths)]
        )
    else:
        comparison_df = pd.concat(
            [read_epoch_summary_files_to_df(path, model_names[i], n_runs, n_epochs) for i, path in
             enumerate(summary_paths)]
        )

    plot_summary(comparison_df, title, xlabel, ylabel, y_log_scale, loss)
