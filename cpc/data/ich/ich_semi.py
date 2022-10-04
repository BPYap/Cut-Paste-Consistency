from typing import Optional

from cpc.data.hybrid_data import HybridDataModule
from cpc.data.ich.ich import ICHDataModuleBase, ICHDataModule
from cpc.data.image_dataset import ImageDataset


class _ICHDataModuleUnlabeled(ICHDataModuleBase):
    def __init__(self, image_dir, **kwargs):
        super().__init__(**kwargs)

        self.image_dir = image_dir

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = ImageDataset(
            image_dir=self.image_dir,
            image_transform=self.image_transform_train
        )


class ICHDataModuleSemi(HybridDataModule):
    args_schema = {
        **ICHDataModule.args_schema,
        "batch_split": (float, 0.4, "Proportion of unlabeled samples in each mini-batch.")
    }

    def __init__(self, batch_split, **kwargs):
        hparams = {
            "batch_split": batch_split,
        }
        self.save_hyperparameters(hparams)

        batch_size = kwargs["batch_size"]
        unlabeled_batch_size = int(batch_size * batch_split)
        labeled_batch_size = batch_size - unlabeled_batch_size
        del kwargs["batch_size"]
        labeled_dm = ICHDataModule(batch_size=labeled_batch_size, **kwargs)
        del kwargs["data_dir"]
        del kwargs["labeled_split"]
        unlabeled_dm = _ICHDataModuleUnlabeled(labeled_dm.unlabeled_img_dir, batch_size=unlabeled_batch_size, **kwargs)

        super().__init__(labeled_dm, [unlabeled_dm], merge_samples=False)
