import json
import os
import random
import shutil
from typing import Optional

import pandas as pd

from cpc.data.idrid.idrid import IDRiDDataModuleBase
from cpc.data.synth_data import SynthDataset
from cpc.image_utils import color_match

IGNORE_LIST = {'IDRiD_012', 'IDRiD_020', 'IDRiD_027', 'IDRiD_028', 'IDRiD_029', 'IDRiD_048', 'IDRiD_053',
               'IDRiD_057', 'IDRiD_058', 'IDRiD_059', 'IDRiD_064', 'IDRiD_066', 'IDRiD_073', 'IDRiD_082',
               'IDRiD_083', 'IDRiD_091', 'IDRiD_102', 'IDRiD_128', 'IDRiD_132', 'IDRiD_185', 'IDRiD_186',
               'IDRiD_196', 'IDRiD_207', 'IDRiD_224', 'IDRiD_231', 'IDRiD_232', 'IDRiD_236', 'IDRiD_237',
               'IDRiD_239', 'IDRiD_242', 'IDRiD_245', 'IDRiD_246', 'IDRiD_247', 'IDRiD_248', 'IDRiD_254',
               'IDRiD_261', 'IDRiD_263', 'IDRiD_271', 'IDRiD_277', 'IDRiD_278', 'IDRiD_280', 'IDRiD_281',
               'IDRiD_282', 'IDRiD_283', 'IDRiD_284', 'IDRiD_294', 'IDRiD_299', 'IDRiD_300', 'IDRiD_306',
               'IDRiD_308', 'IDRiD_309', 'IDRiD_310', 'IDRiD_311', 'IDRiD_313', 'IDRiD_314', 'IDRiD_320',
               'IDRiD_324', 'IDRiD_326', 'IDRiD_328', 'IDRiD_329', 'IDRiD_330', 'IDRiD_347', 'IDRiD_354',
               'IDRiD_356', 'IDRiD_366', 'IDRiD_369', 'IDRiD_380'}


class IDRiDDatasetSynth(SynthDataset):
    def __init__(self, task_id,
                 foreground_dir, mask_dir, background_dir,
                 mask_blur, background_blur, image_transform, mask_transform,
                 matched_pairs=None, return_background=True, background_mask_dir=None):
        super().__init__(
            foreground_dir, mask_dir, background_dir,
            mask_blur, background_blur, image_transform, mask_transform,
            matched_pairs, return_background, background_mask_dir
        )
        self.task_id = task_id

    def get_mask_filename(self, image_filename):
        return f"{image_filename.split('.')[0]}_{self.task_id}.tif"


class IDRiDDataModuleSynth(IDRiDDataModuleBase):
    def __init__(self, num_samples, img_match, mask_blur, background_blur,
                 return_background=True, background_mask_dir=None, **kwargs):
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
                |- B. Disease Grading
                    |- 1. Original Images
                        |- a. Training Set
                    |- 2. Groundtruths
                        |- a. IDRiD_Disease Grading_Training Labels.csv
        """
        super().__init__(**kwargs)

        self.num_samples = num_samples
        self.img_match = img_match
        self.mask_blur = mask_blur
        self.background_blur = background_blur
        self.return_background = return_background
        self.background_mask_dir = background_mask_dir

        # directories of filtered backgrounds
        self.filtered_background_dir = os.path.join(self.data_dir, f"filtered_background_{num_samples}")

        # path to cached color matching file
        self.color_match_cache = os.path.join(self.data_dir, f"color_match_{num_samples}_{self.task_id}.json")

        # directories of foreground images
        foreground_root_dir = os.path.join(self.data_dir, "A. Segmentation")
        task_dir = IDRiDDataModuleBase.MASK_DIR[self.task_id]
        self.foreground_dir = os.path.join(
            foreground_root_dir,
            "1. Original Images",
            "train_split",
            task_dir
        )
        self.mask_dir = os.path.join(
            foreground_root_dir,
            "2. All Segmentation Groundtruths",
            "train_split",
            task_dir
        )

        # directories of background images
        background_root_dir = os.path.join(self.data_dir, "B. Disease Grading")
        self.background_dir = os.path.join(
            background_root_dir,
            "1. Original Images",
            "a. Training Set"
        )
        self.background_csv_path = os.path.join(
            background_root_dir,
            "2. Groundtruths",
            "a. IDRiD_Disease Grading_Training Labels.csv"
        )

    def prepare_data(self):
        # store attributes in local variables for convenience
        filtered_background_dir = self.filtered_background_dir
        color_match_cache = self.color_match_cache
        foreground_dir = self.foreground_dir
        background_dir = self.background_dir
        background_csv_path = self.background_csv_path
        num_samples = self.num_samples
        img_match = self.img_match

        # filter backgrounds
        if not os.path.exists(filtered_background_dir):
            os.makedirs(filtered_background_dir)
            background_ids = []
            for _, row in pd.read_csv(background_csv_path).iterrows():
                image_id = row["Image name"]
                if image_id not in IGNORE_LIST:  # filter samples in `IGNORE_LIST`
                    background_ids.append(image_id)
            for image_id in random.sample(background_ids, num_samples):
                shutil.copy(os.path.join(background_dir, f"{image_id}.jpg"), filtered_background_dir)

        # perform color matching
        if img_match and not os.path.exists(color_match_cache):
            with open(color_match_cache, 'w') as f:
                matched_pairs = color_match(filtered_background_dir, foreground_dir)
                json.dump(matched_pairs, f)

    def setup(self, stage: Optional[str] = None):
        if self.img_match:
            with open(self.color_match_cache, 'r') as f:
                matched_pairs = json.load(f)
        else:
            matched_pairs = None

        self.dataset_train = IDRiDDatasetSynth(
            foreground_dir=self.foreground_dir,
            mask_dir=self.mask_dir,
            background_dir=self.filtered_background_dir,
            task_id=self.task_id,
            mask_blur=self.mask_blur,
            background_blur=self.background_blur,
            image_transform=self.image_transform_train,
            mask_transform=self.mask_transform_train,
            matched_pairs=matched_pairs,
            return_background=self.return_background,
            background_mask_dir=self.background_mask_dir,
        )
