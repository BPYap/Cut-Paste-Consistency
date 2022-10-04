import json
import os
import shutil
from typing import Optional

import numpy as np
from PIL import Image

from cpc.data.ich.ich import ICHDataModule
from cpc.data.synth_data import SynthDataset
from cpc.image_utils import grayscale_match


class ICHDataModuleSynth(ICHDataModule):
    def __init__(self, img_match, mask_blur, background_blur,
                 return_background=True, background_mask_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.img_match = img_match
        self.mask_blur = mask_blur
        self.background_blur = background_blur
        self.return_background = return_background
        self.background_mask_dir = background_mask_dir

        self.foreground_dir = os.path.join(os.path.dirname(self.train_img_dir), "foreground")
        self.mask_dir = self.train_mask_dir
        self.background_dir = self.unlabeled_img_dir

        # path to cached color matching file
        self.color_match_cache = os.path.join(os.path.dirname(self.train_img_dir), "color_match.json")

    def prepare_data(self):
        # store attributes in local variables for convenience
        color_match_cache = self.color_match_cache
        foreground_dir = self.foreground_dir
        mask_dir = self.mask_dir
        background_dir = self.background_dir
        img_match = self.img_match

        # extract foregrounds
        if not os.path.exists(foreground_dir):
            os.makedirs(foreground_dir)
            # keep slices with non-empty masks
            for filename in os.listdir(self.train_img_dir):
                mask_path = os.path.join(mask_dir, filename)
                mask = Image.open(mask_path)
                if np.asarray(mask).sum() > 0:
                    shutil.copy(os.path.join(self.train_img_dir, filename), foreground_dir)

        # perform color matching
        if img_match and not os.path.exists(color_match_cache):
            with open(color_match_cache, 'w') as f:
                matched_pairs = grayscale_match(background_dir, foreground_dir)
                json.dump(matched_pairs, f)

    def setup(self, stage: Optional[str] = None):
        if self.img_match:
            with open(self.color_match_cache, 'r') as f:
                matched_pairs = json.load(f)
        else:
            matched_pairs = None

        self.dataset_train = SynthDataset(
            foreground_dir=self.foreground_dir,
            mask_dir=self.mask_dir,
            background_dir=self.background_dir,
            mask_blur=self.mask_blur,
            background_blur=self.background_blur,
            image_transform=self.image_transform_train,
            mask_transform=self.mask_transform_train,
            matched_pairs=matched_pairs,
            return_background=self.return_background,
            background_mask_dir=self.background_mask_dir,
            trim_threshold=1
        )
