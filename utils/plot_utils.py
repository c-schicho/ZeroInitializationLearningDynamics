import os
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame

from utils.summary_writer import read_summary_files_to_df


def plot_summary(summary_df: DataFrame, title: str):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    n_epochs = max(list(summary_df.step))
    sns.set_theme(palette="colorblind", style="whitegrid")
    plot = sns.lineplot(x="step", y="value", hue="model", data=summary_df)
    plot.set(title=title, xlabel="Epoch", ylabel="Accuracy")
    plot.set_xlim(0, n_epochs)
    plt.legend(loc="lower right")


def plot_comparison(
        summary_paths: List[str],
        n_runs: int,
        n_epochs: int,
        title: Union[str, None] = None,
        model_names: Union[List[str], None] = None
):
    if model_names is None:
        model_names = [os.path.split(path)[-1] for path in summary_paths]

    if title is None:
        title = "Model comparison"

    if len(summary_paths) != len(model_names):
        raise ValueError("summary_paths and model_names must have the same size")

    comparison_df = pd.concat(
        [read_summary_files_to_df(path, model_names[i], n_runs, n_epochs) for i, path in enumerate(summary_paths)]
    )

    plot_summary(comparison_df, title)
