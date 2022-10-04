import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT = transforms.Compose([
    transforms.ToTensor()
])


def transform_foreground(foreground, mask):
    if mask.mode == 'RGBA':
        mask = mask.split()[0]  # extract mask from the red channel
    mask = mask.point(lambda p: int(p != 0), mode='1')

    geometry_jitter = transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1))
    foreground.putalpha(mask)
    foreground = geometry_jitter(foreground)
    mask = foreground.getchannel('A').point(lambda p: 1 if p != 0 else 0, mode='1')

    color_jitter = transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.8
    )
    foreground = color_jitter(foreground.convert('RGB'))

    return foreground, mask


def blur_background(background, mode='gaussian'):
    if mode == 'random':
        mode = random.choice(['none', 'gaussian'])
    if mode == 'none':
        pass
    elif mode == 'gaussian':
        background = background.filter(filter=ImageFilter.GaussianBlur(3))
    else:
        raise ValueError(f"Unknown `mode` `{mode}`. Choose from ['none', 'gaussian', 'random'].")

    return background


class ToMask:
    def __call__(self, image):
        if image.mode == 'P':
            np_image = np.array(image)
            if np_image.ndim == 2:
                np_image = np_image[:, :, None]

            tensor = torch.from_numpy(np_image.transpose((2, 0, 1)))
        else:
            tensor = transforms.functional.to_tensor(image)

        return tensor.squeeze().long()


def crop_transform(size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img_transform = transforms.Compose([
        transforms.RandomCrop(size),
        transforms.RandomRotation(90),
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ]),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mask_transform = transforms.Compose(img_transform.transforms[:3] + [ToMask()])

    return img_transform, mask_transform


def resize_transform(size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomRotation(90),
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ]),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mask_transform = transforms.Compose(img_transform.transforms[:3] + [ToMask()])

    return img_transform, mask_transform


def eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img_transform = transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mask_transform = ToMask()

    return img_transform, mask_transform
