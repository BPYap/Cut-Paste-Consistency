from typing import Optional

import torch
from pytorch_lightning import LightningDataModule


class HybridDataLoader:
    def __init__(self, main_loader, sub_loaders, merge_samples):
        self.main_loader = main_loader
        self.main_iterator = None
        self.sub_loaders = sub_loaders
        self.sub_iterators = [iter(loader) for loader in sub_loaders]
        self.merge_samples = merge_samples

    def __len__(self):
        return len(self.main_loader)

    def __iter__(self):
        self.main_iterator = iter(self.main_loader)

        return self

    def __next__(self):
        batches = [next(self.main_iterator)]
        for i in range(len(self.sub_iterators)):
            try:
                batch = next(self.sub_iterators[i])
            except StopIteration:
                self.sub_iterators[i] = iter(self.sub_loaders[i])
                batch = next(self.sub_iterators[i])
            batches.append(batch)

        if self.merge_samples:
            merged = []
            for i in range(len(batches[0])):
                merged.append(torch.cat([batch[i] for batch in batches]))

            return tuple(merged)
        else:
            return tuple(batches)


class HybridDataModule(LightningDataModule):
    def __init__(self, main_dm, sub_dms, merge_samples):
        super().__init__()
        self.main_dm = main_dm
        self.sub_dms = sub_dms
        self.merge_samples = merge_samples

    @property
    def num_classes(self):
        return self.main_dm.num_classes

    def prepare_data(self):
        self.main_dm.prepare_data()
        for dm in self.sub_dms:
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.main_dm.setup()
        for dm in self.sub_dms:
            dm.setup()

    def train_dataloader(self, drop_last=True):
        sub_loaders = [dm.train_dataloader() for dm in self.sub_dms if dm.batch_size]
        loader = HybridDataLoader(self.main_dm.train_dataloader(), sub_loaders, self.merge_samples)

        return loader

    def val_dataloader(self):
        return self.main_dm.val_dataloader()

    def test_dataloader(self):
        return self.main_dm.test_dataloader()
