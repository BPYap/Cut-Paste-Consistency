import math
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from cpc.model.unet import BinaryUNet
from cpc.train_utils import calculate_train_steps


class BinaryUNetCutMix(BinaryUNet):
    args_schema = {
        **BinaryUNet.args_schema,
        "mean_teacher": (bool, False, "Use mean teacher model for pseudo-labelling."),
        "base_ema": (float, 0.996, "Exponential moving average parameter for the teacher network."),
        "alpha": (float, 1.0, "Parameter for the Beta distribution used for bounding box sampling."),
        "unlabeled_weight": (float, 0.01, "Weightage given to the unlabeled term.")
    }

    def __init__(self, mean_teacher, base_ema, alpha, unlabeled_weight, **kwargs):
        super().__init__(**kwargs)

        self.base_ema = base_ema
        self.alpha = alpha
        self.unlabeled_weight = unlabeled_weight

        # teacher network
        if mean_teacher:
            self.teacher_network = deepcopy(self)
            self.teacher_network.requires_grad_(False)
        else:
            self.teacher_network = None

        self.total_steps = ...

    def on_train_start(self):
        _, self.total_steps = calculate_train_steps(self)

    @staticmethod
    def _cutmix(images, preds, alpha):
        # Codes adapted from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
        def rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2

        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(images.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        preds[:, bbx1:bbx2, bby1:bby2] = preds[rand_index, bbx1:bbx2, bby1:bby2]

        return images, preds

    def training_step(self, batch, batch_idx):
        (labeled, masks), unlabeled = batch

        # generate CutMix samples
        with torch.no_grad():
            inference_model = self.teacher_network if self.teacher_network is not None else self
            preds = torch.sigmoid(inference_model(unlabeled)[-1].squeeze(1))
            mixed_images, mixed_targets = self._cutmix(unlabeled, preds, self.alpha)

        pos_weight = torch.tensor([self.pos_weight]).to(self.device)
        logits = self(torch.cat([labeled, mixed_images]))[-1].squeeze(1)

        labeled_logits = logits[:len(labeled)]
        labeled_seg_loss = F.binary_cross_entropy_with_logits(
            labeled_logits, masks.float(), pos_weight=pos_weight
        )
        self.log("train/labeled_seg_loss", labeled_seg_loss)

        mixed_logits = logits[len(labeled):]
        mixed_preds = torch.sigmoid(mixed_logits)
        unlabeled_const_loss = F.mse_loss(mixed_preds, mixed_targets)
        unlabeled_const_loss *= self.unlabeled_weight
        self.log("train/unlabeled_const_loss", unlabeled_const_loss)

        loss = labeled_seg_loss + unlabeled_const_loss
        self.log("train/loss", loss)

        return loss

    def _get_ema(self):
        current_step = self.global_step

        return 1 - (1 - self.base_ema) * (math.cos((current_step * math.pi) / self.total_steps) + 1) / 2

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.teacher_network is None:
            return

        ema = self._get_ema()
        self.log("ema", ema, rank_zero_only=True)

        # update teacher network
        for student_param, teacher_param in zip(self.parameters(), self.teacher_network.parameters()):
            teacher_param.data = ema * teacher_param.data + (1 - ema) * student_param.data
