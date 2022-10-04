import platform
from typing import Optional
from warnings import warn

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    name = "base"
    args_schema = {
        "num_workers": (int, 8, "Number of workers (per process) for data loaders."),
        "batch_size": (int, 32, "Batch size (per process) for data loaders.")
    }

    def __init__(self, num_workers, batch_size, **kwargs):
        super().__init__(**kwargs)
        hparams = {
            "batch_size": batch_size
        }
        self.save_hyperparameters(hparams)

        if num_workers and platform.system() == "Windows":
            # see: https://stackoverflow.com/a/59680818
            warn(
                f"You have requested num_workers={num_workers} on Windows,"
                " but currently recommended is 0, so we set it for you"
            )
            num_workers = 0
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.dataset_train = ...
        self.dataset_val = None
        self.dataset_test = None

    @property
    def num_classes(self):
        return -1

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to be done only from a single process in
        distributed settings"""
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        """Data operations to perform on every GPU."""
        raise NotImplementedError

    def train_dataloader(self, drop_last=True):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self):
        loader = None
        if self.dataset_val:
            loader = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
            )

        return loader

    def test_dataloader(self):
        loader = None
        if self.dataset_test:
            loader = DataLoader(
                self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
            )

        return loader
