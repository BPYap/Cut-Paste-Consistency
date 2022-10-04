import math
import os
import random

import colour
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms.functional import pad, resize
from tqdm import tqdm


def resize_images(image_dir, size):
    """Resize the shorter side of all images in `image_dir` to `size` pixels and save the
    resized images to `image_dir`-`size`.
    """
    new_image_dir = f"{image_dir}-{size}"
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
    else:
        return

    count = 1
    listing = os.listdir(image_dir)
    for file in listing:
        print(f"\rResizing images from '{image_dir}' ({count}/{len(listing)})", end='', flush=True)
        try:
            image = Image.open(os.path.join(image_dir, file))
            new_image = transforms.functional.resize(image, size)
            new_image.save(os.path.join(new_image_dir, file))
        except UnidentifiedImageError:
            continue
        count += 1
    print(flush=True)


def color_match(anchor_img_dir, source_img_dir, resize_factor=0.2):
    matched_pairs = {}
    for anchor_image_file in tqdm(os.listdir(anchor_img_dir), desc="matching"):
        anchor = cv2.imread(os.path.join(anchor_img_dir, anchor_image_file))
        anchor = cv2.resize(anchor, None, fx=resize_factor, fy=resize_factor)
        anchor_lab = cv2.cvtColor(anchor.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        source_image_files = []
        match_scores = []
        for image_file in os.listdir(source_img_dir):
            image = cv2.imread(os.path.join(source_img_dir, image_file))
            image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
            image_lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)

            delta_e = colour.delta_E(anchor_lab, image_lab).mean()
            match_score = 1 / delta_e
            source_image_files.append(image_file)
            match_scores.append(match_score)

        matched_pairs[anchor_image_file] = {"filenames": source_image_files, "scores": match_scores}

    return matched_pairs


def grayscale_match(anchor_img_dir, source_img_dir, resize_factor=0.2):
    matched_pairs = {}
    for anchor_image_file in tqdm(os.listdir(anchor_img_dir), desc="matching"):
        anchor = cv2.imread(os.path.join(anchor_img_dir, anchor_image_file))
        anchor = cv2.resize(anchor, None, fx=resize_factor, fy=resize_factor).astype(np.float32) / 255
        source_image_files = []
        match_scores = []
        for image_file in os.listdir(source_img_dir):
            image = cv2.imread(os.path.join(source_img_dir, image_file))
            image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor).astype(np.float32) / 255

            l1_norm = cv2.norm(anchor - image, cv2.NORM_L2)
            match_score = 1 / l1_norm
            source_image_files.append(image_file)
            match_scores.append(match_score)

        matched_pairs[anchor_image_file] = {"filenames": source_image_files, "scores": match_scores}

    return matched_pairs


def compose_samples(foreground, mask, background, mask_blur):
    mask = mask.point(lambda p: 255 if p != 0 else 0, mode='L')

    # apply blurring to mask
    if mask_blur == 'random':
        mask_blur = random.choice(['none', 'gaussian'])
    if mask_blur == 'none':
        filtered_mask = mask
    elif mask_blur == 'gaussian':
        filtered_mask = mask.filter(filter=ImageFilter.GaussianBlur(3))
    else:
        raise ValueError(f"Unknown `mask_blur` option `{mask_blur}`. Choose from ['none', 'gaussian', 'random'].")

    return Image.composite(foreground, background, filtered_mask), mask.point(lambda p: int(p != 0), mode='1')


def trim_background(background, image, mask, threshold):
    grayscale_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    valid_mask = (grayscale_transform(background) >= (threshold / 255))
    img_tensor = transforms.functional.to_tensor(image) * valid_mask
    mask_tensor = transforms.functional.to_tensor(mask) * valid_mask
    new_image = transforms.functional.to_pil_image(img_tensor)
    new_mask = transforms.functional.to_pil_image(mask_tensor).point(lambda p: int(p != 0), mode='1')

    return new_image, new_mask


def crop_inference(model, inputs, window_size):
    # perform patch-wise inference on the whole image in a sliding-window fashion
    with torch.no_grad():
        batch_size = inputs.shape[0]
        height = inputs.shape[2]
        width = inputs.shape[3]

        stride = window_size // 2
        height_padding = (math.ceil(height / stride) * stride - height) // 2
        width_padding = (math.ceil(width / stride) * stride - width) // 2

        x = torch.cat([pad(im, padding=[width_padding, height_padding]).unsqueeze(dim=0) for im in inputs], dim=0)
        buffer = torch.zeros(batch_size, height + 2 * height_padding, width + 2 * width_padding).to(x.device)
        counter = torch.zeros_like(buffer)
        for r in range(0, buffer.shape[1] - stride, stride):
            for c in range(0, buffer.shape[2] - stride, stride):
                end_r = r + window_size
                end_c = c + window_size
                patches = x[:, :, r: end_r, c: end_c]
                buffer[:, r: end_r, c: end_c] += model(patches)[-1].squeeze(dim=1)
                counter[:, r: end_r, c:end_c] += 1

        y_hat = (buffer / counter)[:, height_padding: height_padding + height, width_padding: width_padding + width]
        y_hat = torch.sigmoid(y_hat).view(batch_size, height, width)

    return y_hat


def resize_inference(model, inputs, image_size):
    # perform full-image inference at lower resolution before resizing back to the original resolution
    with torch.no_grad():
        original_height = inputs.shape[2]
        original_width = inputs.shape[3]

        inputs = resize(inputs, image_size, antialias=True)
        y_hat = torch.sigmoid(model(inputs)[-1].squeeze(dim=1))
        y_hat = resize(y_hat, (original_height, original_width), antialias=True)

    return y_hat
