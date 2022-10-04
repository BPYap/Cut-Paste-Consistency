import os
import random
import shutil
from typing import Optional

import torch
from PIL import Image

from cpc.config import _list_or_int
from cpc.data.base_data import BaseDataModule
from cpc.transforms import crop_transform, resize_transform, eval_transform


class IDRiDDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, task_id, image_transform, mask_transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_ids = []
        self.mask_suffix = task_id
        for filename in os.listdir(mask_dir):
            tokens = filename.split('.')[0].split('_')
            assert task_id == tokens[-1], f"Mask prefix '{tokens[-1]}' does not match the task id '{task_id}'."
            image_id = "_".join(tokens[:2])
            self.image_ids.append(image_id)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, f"{image_id}.jpg"))
        mask = Image.open(os.path.join(self.mask_dir, f"{image_id}_{self.mask_suffix}.tif"))

        if mask.mode == 'RGBA':
            mask = mask.split()[0]  # extract mask from the red channel

        count = 1
        while count <= 100:
            # same seed to ensure random transformations are applied consistently on both image and target
            seed = random.randint(0, 2147483647)

            random.seed(seed)
            torch.manual_seed(seed)
            image_tensor = self.image_transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            mask_tensor = self.mask_transform(mask)

            if mask_tensor.sum() > 0:
                break
            else:
                # keep sampling if mask is empty
                count += 1

        return image_tensor, mask_tensor


class IDRiDDataModuleBase(BaseDataModule):
    name = "IDRiD"
    source = "https://idrid.grand-challenge.org/"
    args_schema = {
        **BaseDataModule.args_schema,
        "data_dir": (str, None, "Data directory."),
        "task_id": (str, "MA", "Name of the segmentation task. Choose from ['MA', 'HE', 'EX', 'SE', 'OD']"),
        "preprocess": (str, "crop", "Preprocess mode. Choose from ['crop', 'resize']."),
        "size": (_list_or_int, 640, "Crop/downscale size to be applied to the whole image."),
    }
    MEAN = (0.457, 0.221, 0.064)
    STD = (0.323, 0.168, 0.087)
    MASK_DIR = {
        'MA': "1. Microaneurysms",
        'HE': "2. Haemorrhages",
        'EX': "3. Hard Exudates",
        'SE': "4. Soft Exudates",
        'OD': "5. Optic Disc"
    }

    def __init__(self, data_dir, task_id, preprocess, size, **kwargs):
        """
        Directory structure of `data_dir`:
            `data_dir`
                |- A. Segmentation
                    |- 1. Original Images
                        |- a. Training Set
                        |- b. Testing Set
                    |- 2. All Segmentation Groundtruths
                        |- a. Training Set
                            |- 1. Microaneurysms
                            |- 2. Haemorrhages
                            |- 3. Hard Exudates
                            |- 4. Soft Exudates
                            |- 5. Optic Disc
                        |- b. Testing Set
                            |- 1. Microaneurysms
                            |- 2. Haemorrhages
                            |- 3. Hard Exudates
                            |- 4. Soft Exudates
                            |- 5. Optic Disc
        """
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.task_id = task_id

        transform_function = crop_transform if preprocess == 'crop' else resize_transform
        self.image_transform_train, self.mask_transform_train = transform_function(
            size=size,
            mean=IDRiDDataModuleBase.MEAN,
            std=IDRiDDataModuleBase.STD
        )
        self.image_transform_eval, self.mask_transform_eval = eval_transform(
            mean=IDRiDDataModuleBase.MEAN,
            std=IDRiDDataModuleBase.STD
        )

    @property
    def num_classes(self):
        return 2

    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError


class IDRiDDataModule(IDRiDDataModuleBase):
    args_schema = {
        **IDRiDDataModuleBase.args_schema,
        "val_split": (float, 0.1, "Proportion of train set to be used as validation set.")
    }

    def __init__(self, val_split, **kwargs):
        super().__init__(**kwargs)

        self.val_split = val_split

        root_dir = os.path.join(self.data_dir, "A. Segmentation")
        task_dir = IDRiDDataModuleBase.MASK_DIR[self.task_id]
        self.train_img_dir = os.path.join(root_dir, "1. Original Images", "train_split", task_dir)
        self.train_mask_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "train_split", task_dir)
        self.val_img_dir = os.path.join(root_dir, "1. Original Images", "val_split", task_dir)
        self.val_mask_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "val_split", task_dir)
        self.test_img_dir = os.path.join(root_dir, "1. Original Images", "b. Testing Set")
        self.test_mask_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "b. Testing Set", task_dir)

    def prepare_data(self):
        # prepare train-val splits
        for dir_ in [self.train_img_dir, self.train_mask_dir, self.val_img_dir, self.val_mask_dir]:
            if os.path.isdir(dir_):
                return
            os.makedirs(dir_)
        root_dir = os.path.join(self.data_dir, "A. Segmentation")
        task_dir = IDRiDDataModuleBase.MASK_DIR[self.task_id]
        img_dir = os.path.join(root_dir, "1. Original Images", "a. Training Set")
        mask_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "a. Training Set", task_dir)
        filenames = os.listdir(mask_dir)
        random.shuffle(filenames)
        len_val = int(self.val_split * len(filenames))
        for _ in range(len_val):
            filename = filenames.pop()
            image_id = "_".join(filename.split("_")[:2])
            shutil.copy(os.path.join(img_dir, f"{image_id}.jpg"), self.val_img_dir)
            shutil.copy(os.path.join(mask_dir, filename), self.val_mask_dir)
        while filenames:
            filename = filenames.pop()
            image_id = "_".join(filename.split("_")[:2])
            shutil.copy(os.path.join(img_dir, f"{image_id}.jpg"), self.train_img_dir)
            shutil.copy(os.path.join(mask_dir, filename), self.train_mask_dir)

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = IDRiDDataset(
            image_dir=self.train_img_dir,
            mask_dir=self.train_mask_dir,
            task_id=self.task_id,
            image_transform=self.image_transform_train,
            mask_transform=self.mask_transform_train
        )

        self.dataset_val = IDRiDDataset(
            image_dir=self.val_img_dir,
            mask_dir=self.val_mask_dir,
            task_id=self.task_id,
            image_transform=self.image_transform_eval,
            mask_transform=self.mask_transform_eval
        )

        self.dataset_test = IDRiDDataset(
            image_dir=self.test_img_dir,
            mask_dir=self.test_mask_dir,
            task_id=self.task_id,
            image_transform=self.image_transform_eval,
            mask_transform=self.mask_transform_eval
        )
