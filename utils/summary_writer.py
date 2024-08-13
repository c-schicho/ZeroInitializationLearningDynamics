import pandas as pd
from tbparse import SummaryReader
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def write_train_summary(writer: SummaryWriter, model, loss: Tensor, global_step: int, param_grad: bool = False):
    writer.add_scalar(tag="train loss", scalar_value=loss.cpu().item(), global_step=global_step)

    if param_grad:
        for name, param in model.named_parameters():
            writer.add_histogram(tag=f"parameter {name}", values=param.cpu(), global_step=global_step)
            writer.add_histogram(tag=f"gradient of parameter {name}", values=param.grad.cpu(), global_step=global_step)


def write_validation_summary(writer: SummaryWriter, loss: float, accuracy: float, global_step: int):
    writer.add_scalar(tag="validation loss", scalar_value=loss, global_step=global_step)
    writer.add_scalar(tag="validation accuracy", scalar_value=accuracy, global_step=global_step)


def write_test_summary(writer: SummaryWriter, loss: float, accuracy: float, global_step: int):
    writer.add_scalar(tag="test loss", scalar_value=loss, global_step=global_step)
    writer.add_scalar(tag="test accuracy", scalar_value=accuracy, global_step=global_step)


def read_epoch_summary_files_to_df(
        summary_path: str,
        model_name: str,
        n_runs: int,
        n_epochs: int,
        tag: str = "test accuracy"
) -> pd.DataFrame:
    reader = SummaryReader(summary_path)
    df = reader.scalars
    df = df[df.tag == tag]
    run_list = list(range(1, n_runs + 1)) * n_epochs
    df["run"] = run_list
    df["model"] = model_name
    return df


def read_update_step_summary_files_to_df(
        summary_path: str,
        model_name: str,
        n_runs: int,
        n_epochs: int,
        tag: str = "train loss"
) -> pd.DataFrame:
    reader = SummaryReader(summary_path)
    df = reader.scalars
    df = df[(df.tag == tag) & (df.step <= 700)]
    run_list = list(range(1, n_runs + 1)) * 700
    df["run"] = run_list
    df["model"] = model_name
    return df
