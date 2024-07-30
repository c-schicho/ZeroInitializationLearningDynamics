import copy
import os

from torch.utils.tensorboard import SummaryWriter

from config import ExperimentConfig
from train.trainer import Trainer
from utils import read_epoch_summary_files_to_df, plot_summary


def run_experiments(model, config: ExperimentConfig):
    summary_path = os.path.join("results", config.model_name)

    print(f"### Running experiment {config.model_name} ###")

    for run in range(1, config.runs + 1):
        summary_writer = SummaryWriter(summary_path)
        run_model = copy.deepcopy(model)

        if config.runs > 1:
            print(f"### {run}. Run ###")

        trainer = Trainer(model=run_model, lr=config.lr, writer=summary_writer)
        trainer.train(
            train_loader=config.train_loader,
            val_loader=config.val_loader,
            test_loader=config.test_loader,
            num_epochs=config.epochs,
            train_summary=config.train_summary,
            test_after_epoch=config.test_after_epoch
        )

    summary_df = read_epoch_summary_files_to_df(
        summary_path=summary_path,
        model_name=config.model_name,
        n_runs=config.runs,
        n_epochs=config.epochs
    )
    plot_summary(summary_df, title=f"Experiments for {config.model_name}")
