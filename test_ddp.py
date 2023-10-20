import pytorch_lightning as pl
import torch
from sklearn.datasets import load_iris
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class IrisDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        iris = load_iris()
        self.data = TensorDataset(
            torch.tensor(
                iris.data, dtype=torch.float32), torch.tensor(
                iris.target))

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=32, num_workers=16)


class LogisticRegression(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    model = LogisticRegression()
    data = IrisDataModule()

    trainer = pl.Trainer(
        max_epochs=10,
        devices=2,
        accelerator='cuda',
        strategy="ddp",
        log_every_n_steps=1,
    )
    trainer.fit(model, data)
