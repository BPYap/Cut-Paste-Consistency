import os
import random
import shutil
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch
from PIL import Image

from cpc.config import _list_or_int
from cpc.data.base_data import BaseDataModule
from cpc.transforms import crop_transform, resize_transform, eval_transform


def create_unlabel_split(data_dir, labeled_split):
    source_img_dir = os.path.join(data_dir, "train", "image")
    source_mask_dir = os.path.join(data_dir, "train", "label")
    csv_labels = os.path.join(os.path.dirname(data_dir), "hemorrhage_diagnosis_raw_ct.csv")
    labeled_img_dir = os.path.join(data_dir, f"semi-{labeled_split}", "image")
    labeled_mask_dir = os.path.join(data_dir, f"semi-{labeled_split}", "label")
    unlabeled_img_dir = os.path.join(data_dir, f"semi-{labeled_split}", "unlabel")

    if os.path.exists(labeled_img_dir) and os.path.exists(labeled_mask_dir) and os.path.exists(unlabeled_img_dir):
        return labeled_img_dir, labeled_mask_dir, unlabeled_img_dir

    for directory in [labeled_img_dir, labeled_mask_dir, unlabeled_img_dir]:
        try:
            os.removedirs(directory)
        except FileNotFoundError:
            pass
        os.makedirs(directory)

    # get labels for each patient
    patient_ids = {int(filename.split('.')[0].split('_')[0]) for filename in os.listdir(source_img_dir)}
    patient_labels = defaultdict(bool)
    num_slices = defaultdict(int)
    for _, row in pd.read_csv(csv_labels).iterrows():
        patient_id = row['PatientNumber']
        positive = row['No_Hemorrhage'] == 0
        if patient_id not in patient_ids:
            continue
        patient_labels[patient_id] = max(patient_labels[patient_id], positive)
        num_slices[patient_id] += 1

    # organize into positive and negative list
    positive_patients = []
    negative_patients = []
    for patient_id, positive in patient_labels.items():
        if positive:
            positive_patients.append(patient_id)
        else:
            negative_patients.append(patient_id)

    # stratified sampling
    for _list in [positive_patients, negative_patients]:
        num_labeled = int(len(_list) * labeled_split)
        random.shuffle(_list)
        for patient_id in _list[:num_labeled]:
            for slice_id in range(num_slices[patient_id]):
                try:
                    shutil.copy(os.path.join(source_img_dir, f"{patient_id}_{slice_id}.png"), labeled_img_dir)
                    shutil.copy(os.path.join(source_mask_dir, f"{patient_id}_{slice_id}.png"), labeled_mask_dir)
                except FileNotFoundError:
                    continue
        for patient_id in _list[num_labeled:]:
            for slice_id in range(num_slices[patient_id]):
                try:
                    shutil.copy(os.path.join(source_img_dir, f"{patient_id}_{slice_id}.png"), unlabeled_img_dir)
                except FileNotFoundError:
                    continue

    return labeled_img_dir, labeled_mask_dir, unlabeled_img_dir


class ICHDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_transform, mask_transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_ids = [filename.split('.')[0] for filename in os.listdir(mask_dir)]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, f"{image_id}.png"))
        mask = Image.open(os.path.join(self.mask_dir, f"{image_id}.png"))

        # same seed to ensure random transformations are applied consistently on both image and target
        seed = random.randint(0, 2147483647)

        random.seed(seed)
        torch.manual_seed(seed)
        image_tensor = self.image_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        mask_tensor = self.mask_transform(mask)

        return image_tensor, mask_tensor


class ICHDataModuleBase(BaseDataModule):
    name = "CT-ICH"
    source = "https://physionet.org/content/ct-ich/1.3.1/"
    args_schema = {
        **BaseDataModule.args_schema,
        "preprocess": (str, "crop", "Preprocess mode. Choose from ['crop', 'resize']."),
        "size": (_list_or_int, 512, "Crop/downscale size to be applied to the whole image.")
    }
    MEAN = 0.187
    STD = 0.319

    def __init__(self, preprocess, size, **kwargs):
        super().__init__(**kwargs)

        transform_function = crop_transform if preprocess == 'crop' else resize_transform
        self.image_transform_train, self.mask_transform_train = transform_function(
            size=size,
            mean=ICHDataModuleBase.MEAN,
            std=ICHDataModuleBase.STD
        )
        self.image_transform_eval, self.mask_transform_eval = eval_transform(
            mean=ICHDataModuleBase.MEAN,
            std=ICHDataModuleBase.STD
        )

    @property
    def num_classes(self):
        return 2

    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError


class ICHDataModule(ICHDataModuleBase):
    args_schema = {
        **ICHDataModuleBase.args_schema,
        "data_dir": (str, None, "Data directory."),
        "labeled_split": (float, 1.0, "Proportion of samples to be sampled as labeled samples.")
    }

    def __init__(self, data_dir, labeled_split, **kwargs):
        """
        Directory structure of `data_dir`:
            `data_dir`
                |- train
                    |- image
                    |- label
                |- test
                    |- image
                    |- label
        """
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.labeled_split = labeled_split

        if labeled_split >= 1:
            self.train_img_dir = os.path.join(data_dir, "train", "image")
            self.train_mask_dir = os.path.join(data_dir, "train", "label")
            self.unlabeled_img_dir = None
        else:
            self.train_img_dir, self.train_mask_dir, self.unlabeled_img_dir = \
                create_unlabel_split(data_dir, labeled_split)

        self.test_img_dir = os.path.join(data_dir, "test", "image")
        self.test_mask_dir = os.path.join(data_dir, "test", "label")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = ICHDataset(
            image_dir=self.train_img_dir,
            mask_dir=self.train_mask_dir,
            image_transform=self.image_transform_train,
            mask_transform=self.mask_transform_train
        )

        self.dataset_test = ICHDataset(
            image_dir=self.test_img_dir,
            mask_dir=self.test_mask_dir,
            image_transform=self.image_transform_eval,
            mask_transform=self.mask_transform_eval
        )
