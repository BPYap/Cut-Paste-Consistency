import os

import torch
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_transform):
        self.image_dir = image_dir
        self.image_transform = image_transform

        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(os.path.join(self.image_dir, image_file))
        image_tensor = self.image_transform(image)

        return image_tensor
