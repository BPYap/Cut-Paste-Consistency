import os
from typing import Optional

from cpc.data.hybrid_data import HybridDataModule
from cpc.data.ich.ich import ICHDataset, ICHDataModuleBase, ICHDataModule


class _ICHDataModuleExtra(ICHDataModuleBase):
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)

        self.extra_img_dir = os.path.join(data_dir, "images")
        self.extra_mask_dir = os.path.join(data_dir, "masks")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = ICHDataset(
            image_dir=self.extra_img_dir,
            mask_dir=self.extra_mask_dir,
            image_transform=self.image_transform_train,
            mask_transform=self.mask_transform_train
        )


class ICHDataModuleExtra(HybridDataModule):
    args_schema = {
        **ICHDataModule.args_schema,
        "extra_data_dir": (str, None, "Directory consisting of additional images and masks."),
        "batch_split": (float, 0.4, "Proportion of extra samples in each mini-batch.")
    }

    def __init__(self, extra_data_dir, batch_split, **kwargs):
        hparams = {
            "batch_split": batch_split
        }
        self.save_hyperparameters(hparams)

        batch_size = kwargs["batch_size"]
        extra_batch_size = int(batch_size * batch_split)
        main_batch_size = batch_size - extra_batch_size
        del kwargs["batch_size"]

        self.dm = ICHDataModule(batch_size=main_batch_size, **kwargs)
        del kwargs["data_dir"]
        del kwargs["labeled_split"]
        self.extra_dm = _ICHDataModuleExtra(data_dir=extra_data_dir, batch_size=extra_batch_size, **kwargs)

        super().__init__(self.dm, [self.extra_dm], merge_samples=False)
