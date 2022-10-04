import os
import shutil
from typing import Optional

from torch.utils.data import random_split

from cpc.data.hybrid_data import HybridDataModule
from cpc.data.idrid.idrid import IDRiDDataModuleBase, IDRiDDataModule
from cpc.data.idrid.idrid_synth import IGNORE_LIST
from cpc.data.image_dataset import ImageDataset


class IDRiDDataModuleUnlabeled(IDRiDDataModuleBase):
    def __init__(self, num_samples, **kwargs):
        super().__init__(**kwargs)

        self.original_dir = os.path.join(
            self.data_dir,
            "B. Disease Grading",
            "1. Original Images",
            "a. Training Set"
        )
        self.filtered_dir = os.path.join(self.data_dir, "unlabeled_filtered")
        self.num_samples = num_samples

    def prepare_data(self):
        # filter samples in `IGNORE_LIST`
        if not os.path.exists(self.filtered_dir):
            os.makedirs(self.filtered_dir)
            for image_file in os.listdir(self.original_dir):
                if image_file.split('.')[0] not in IGNORE_LIST:
                    shutil.copy(os.path.join(self.original_dir, image_file), self.filtered_dir)
        else:
            return

    def setup(self, stage: Optional[str] = None):
        dataset = ImageDataset(
            image_dir=self.filtered_dir,
            image_transform=self.image_transform_train
        )
        self.dataset_train = random_split(dataset, [self.num_samples, len(dataset) - self.num_samples])[0]


class IDRiDDataModuleSemi(HybridDataModule):
    args_schema = {
        **IDRiDDataModule.args_schema,
        "batch_split": (float, 0.4, "Proportion of unlabeled samples in each mini-batch."),
        "num_unlabeled": (int, 300, "Number of unlabeled images to sample.")
    }

    def __init__(self, val_split, batch_split, num_unlabeled, **kwargs):
        hparams = {
            "batch_split": batch_split,
            "num_unlabeled": num_unlabeled
        }
        self.save_hyperparameters(hparams)

        batch_size = kwargs["batch_size"]
        unlabeled_batch_size = int(batch_size * batch_split)
        labeled_batch_size = batch_size - unlabeled_batch_size
        del kwargs["batch_size"]
        self.labeled_dm = IDRiDDataModule(val_split, batch_size=labeled_batch_size, **kwargs)
        self.unlabeled_dm = IDRiDDataModuleUnlabeled(num_unlabeled, batch_size=unlabeled_batch_size, **kwargs)

        super().__init__(self.labeled_dm, [self.unlabeled_dm], merge_samples=False)
