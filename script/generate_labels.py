# Generate pseudo-labels for self-training

import argparse
import os
import shutil

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from cpc.data.idrid.idrid import IDRiDDataModuleBase
from cpc.data.ich.ich import ICHDataModule
from cpc.image_utils import crop_inference, resize_inference
from cpc.model import BinaryUNet

DATA_MODULES = {
    "idrid": IDRiDDataModuleBase,
    "ich": ICHDataModule
}
LN_MODULES = {
    "unet": BinaryUNet
}


def main(_args, _dm_cls, _model_cls):
    model_args = {arg: default_value for arg, (_, default_value, _) in _model_cls.args_schema.items()}
    model_args["input_channels"] = _args.input_channels
    model = _model_cls.load_from_checkpoint(_args.pretrain_model_path, **model_args).cuda()

    mode = _args.inference_mode
    size = _args.inference_size
    image_dir = _args.image_dir
    output_dir = _args.output_dir
    mask_ext = _args.mask_ext
    suffix = _args.suffix
    threshold = _args.threshold

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    new_image_dir = os.path.join(output_dir, "images")
    new_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(new_image_dir)
    os.makedirs(new_mask_dir)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_dm_cls.MEAN, _dm_cls.STD)
    ])
    grayscale_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    for filename in tqdm(os.listdir(image_dir), desc="processing"):
        image_id = filename.split('.')[0]
        image = Image.open(os.path.join(image_dir, filename))
        tensor = img_transform(image).cuda()
        valid_mask = (grayscale_transform(image) >= (threshold / 255)).squeeze(0).cuda()
        inference_function = crop_inference if mode == 'crop' else resize_inference
        prediction = inference_function(model, tensor.unsqueeze(0), size).squeeze(0) * valid_mask
        mask = Image.fromarray(np.array(((prediction >= 0.5) * 255).cpu(), dtype=np.uint8), mode='L')
        mask = mask.point(lambda p: int(p != 0), mode='1')

        shutil.copy(os.path.join(image_dir, filename), new_image_dir)
        mask.save(os.path.join(new_mask_dir, f"{image_id}{suffix}.{mask_ext}"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_module', type=str, choices=list(DATA_MODULES.keys()))
    arg_parser.add_argument("model", type=str, choices=list(LN_MODULES.keys()))
    arg_parser.add_argument("--input_channels", type=int, help="Number of input channels.")
    arg_parser.add_argument("--pretrain_model_path", type=str, help="Path to pretrained model.")
    arg_parser.add_argument("--inference_mode", type=str, help="Choose from ['crop', 'resize'].")
    arg_parser.add_argument("--inference_size", type=int, help="Size information for inference.")
    arg_parser.add_argument("--image_dir", type=str, help="Directory consisting of unlabeled images.")
    arg_parser.add_argument("--output_dir", type=str, help="Output directory to store generated labels.")
    arg_parser.add_argument("--mask_ext", type=str, default='tif', help="File extension of generated mask.")
    arg_parser.add_argument("--suffix", type=str, default='', help="Optional suffix to add to the generated filenames.")
    arg_parser.add_argument("--threshold", type=int, default=15, help="Threshold (0 ~ 255) for background pixels.")

    args = arg_parser.parse_args()
    dm_cls = DATA_MODULES[args.data_module]
    model_cls = LN_MODULES[args.model]
    main(args, dm_cls, model_cls)
