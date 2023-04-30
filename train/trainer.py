from typing import Union, Dict

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import write_train_summary, write_test_summary, calculate_accuracy


class Trainer:

    def __init__(self, model, lr: float, writer: SummaryWriter):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.loss_fun = CrossEntropyLoss(reduction="sum")
        self.writer = writer

    def train(
            self,
            train_loader: DataLoader,
            val_loader: Union[DataLoader, None] = None,
            test_loader: Union[DataLoader, None] = None,
            num_epochs: int = 10,
            train_summary: bool = True,
            test_after_epoch: bool = True
    ):
        self.model.train()
        update_step = 1

        for epoch in range(1, num_epochs + 1):
            for data_batch in tqdm(train_loader, total=len(train_loader), ncols=90, desc=f"Epoch {epoch}/{num_epochs}"):
                img_data = data_batch[0].to(self.device)
                targets = data_batch[1].to(self.device)
                batch_size = targets.size(dim=0)

                self.optimizer.zero_grad()
                outputs = self.model(img_data)
                loss = self.loss_fun(outputs, targets)
                loss.backward()
                self.optimizer.step()

                loss /= batch_size  # sample average loss

                if train_summary:
                    write_train_summary(writer=self.writer, model=self.model, loss=loss, global_step=update_step)

                update_step += 1

            if test_after_epoch and test_loader is not None:
                self.__calculate_write_test_metrics(test_loader, epoch)

    def __calculate_write_test_metrics(self, dataloader: DataLoader, step: int):
        results = self.__calculate_metrics(dataloader)
        write_test_summary(writer=self.writer, loss=results["loss"], accuracy=results["accuracy"], global_step=step)

    def __calculate_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data_batch in dataloader:
                img_data = data_batch[0].to(self.device)
                targets = data_batch[1].to(self.device)

                outputs = self.model(img_data)

                all_outputs.extend(outputs)
                all_targets.extend(targets)

            all_outputs = torch.stack(all_outputs)
            all_targets = torch.stack(all_targets)

            loss = self.loss_fun(all_outputs, all_targets)
            accuracy = calculate_accuracy(all_outputs, all_targets)

            loss /= len(dataloader.dataset)

        return {'accuracy': accuracy, 'loss': loss}
