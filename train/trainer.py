from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from typing import Union
from tqdm import tqdm
from utils import write_train_summary


class Trainer:

    def __init__(self, model, lr: float, writer: SummaryWriter):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=lr)
        self.loss_fun = CrossEntropyLoss(reduction="sum")
        self.writer = writer

    def train(
            self,
            train_loader: DataLoader,
            val_loader: Union[DataLoader, None] = None,
            test_loader: Union[DataLoader, None] = None,
            num_epochs: int = 10
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

                write_train_summary(writer=self.writer, model=self.model, loss=loss, global_step=update_step)

                update_step += 1
