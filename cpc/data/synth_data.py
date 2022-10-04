import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from cpc.image_utils import compose_samples, trim_background
from cpc.transforms import transform_foreground, blur_background


class SynthDataset(Dataset):
    def __init__(self, foreground_dir, mask_dir, background_dir,
                 mask_blur, background_blur, image_transform, mask_transform,
                 matched_pairs=None, return_background=True, background_mask_dir=None, trim_threshold=15):
        self.foreground_dir = foreground_dir
        self.mask_dir = mask_dir
        self.background_dir = background_dir
        self.mask_blur = mask_blur
        self.background_blur = background_blur
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.matched_pairs = matched_pairs
        self.return_background = return_background
        self.background_mask_dir = background_mask_dir
        self.trim_threshold = trim_threshold

        self.foreground_filenames = os.listdir(self.foreground_dir)
        self.background_filenames = os.listdir(self.background_dir)

    def __len__(self):
        return len(self.background_filenames)

    def get_mask_filename(self, image_filename):
        return image_filename

    def __getitem__(self, index):
        background_filename = self.background_filenames[index]
        foreground_filenames = []
        if self.matched_pairs:
            filenames = self.matched_pairs[background_filename]['filenames']
            scores = self.matched_pairs[background_filename]['scores']
            top_5 = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)[:5]
            random.shuffle(top_5)
            foreground_filenames.append(top_5.pop(0)[0])
        else:
            foreground_filenames.append(random.choice(self.foreground_filenames))

        background = Image.open(os.path.join(self.background_dir, background_filename))
        synth_image = blur_background(background, self.background_blur)
        masks = []
        for foreground_filename in foreground_filenames:
            foreground = Image.open(os.path.join(self.foreground_dir, foreground_filename))
            mask_filename = self.get_mask_filename(foreground_filename)
            mask = Image.open(os.path.join(self.mask_dir, mask_filename))

            foreground, mask = transform_foreground(foreground, mask)
            synth_image, synth_mask = compose_samples(foreground, mask, synth_image, self.mask_blur)
            synth_image, synth_mask = trim_background(background, synth_image, synth_mask,
                                                      threshold=self.trim_threshold)
            masks.append(synth_mask)

        if self.background_mask_dir:
            background_mask_filename = self.get_mask_filename(background_filename)
            masks.append(Image.open(os.path.join(self.background_mask_dir, background_mask_filename)))

        # same seed to ensure random transformations are applied consistently on both image and target
        seed = random.randint(0, 2147483647)

        random.seed(seed)
        torch.manual_seed(seed)
        synth_image_tensor = self.image_transform(synth_image)

        random.seed(seed)
        torch.manual_seed(seed)
        synth_mask_tensor = self.mask_transform(masks.pop(0))

        while masks:
            random.seed(seed)
            torch.manual_seed(seed)
            synth_mask_tensor += self.mask_transform(masks.pop(0))

        random.seed(seed)
        torch.manual_seed(seed)
        background_tensor = self.image_transform(background)

        return (synth_image_tensor, synth_mask_tensor, background_tensor) if self.return_background \
            else (synth_image_tensor, synth_mask_tensor)
