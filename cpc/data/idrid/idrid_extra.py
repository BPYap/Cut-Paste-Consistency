import os
from typing import Optional

from torch.utils.data import random_split

from cpc.data.hybrid_data import HybridDataModule
from cpc.data.idrid.idrid import IDRiDDataset, IDRiDDataModuleBase, IDRiDDataModule
from cpc.data.idrid.idrid_synth import IGNORE_LIST


class _IDRiDDataModuleExtra(IDRiDDataModuleBase):
    def __init__(self, num_samples, **kwargs):
        super().__init__(**kwargs)

        self.extra_img_dir = os.path.join(self.data_dir, "images")
        self.extra_mask_dir = os.path.join(self.data_dir, "masks")
        self.num_samples = num_samples

    def prepare_data(self):
        # remove samples in `IGNORE_LIST`
        for ignore_id in IGNORE_LIST:
            try:
                os.remove(os.path.join(self.extra_img_dir, f"{ignore_id}.jpg"))
            except FileNotFoundError:
                pass

            try:
                os.remove(os.path.join(self.extra_mask_dir, f"{ignore_id}_{self.task_id}.tif"))
            except FileNotFoundError:
                pass

    def setup(self, stage: Optional[str] = None):
        dataset_extra = IDRiDDataset(
            image_dir=self.extra_img_dir,
            mask_dir=self.extra_mask_dir,
            task_id=self.task_id,
            image_transform=self.image_transform_train,
            mask_transform=self.mask_transform_train
        )
        self.dataset_train = random_split(dataset_extra, [self.num_samples, len(dataset_extra) - self.num_samples])[0]


class IDRiDDataModuleExtra(HybridDataModule):
    args_schema = {
        **IDRiDDataModule.args_schema,
        "extra_data_dir": (str, None, "Directory consisting of additional images and masks."),
        "num_samples": (int, 300, "Number of instances to sample."),
        "batch_split": (float, 0.4, "Proportion of extra samples in each mini-batch.")
    }

    def __init__(self, val_split, extra_data_dir, num_samples, batch_split, **kwargs):
        hparams = {
            "num_samples": num_samples,
            "batch_split": batch_split
        }
        self.save_hyperparameters(hparams)

        batch_size = kwargs["batch_size"]
        extra_batch_size = int(batch_size * batch_split)
        main_batch_size = batch_size - extra_batch_size
        del kwargs["batch_size"]
        self.dm = IDRiDDataModule(val_split, batch_size=main_batch_size, **kwargs)
        del kwargs["data_dir"]
        self.extra_dm = _IDRiDDataModuleExtra(
            num_samples, data_dir=extra_data_dir, batch_size=extra_batch_size, **kwargs
        )

        super().__init__(self.dm, [self.extra_dm], merge_samples=False)
