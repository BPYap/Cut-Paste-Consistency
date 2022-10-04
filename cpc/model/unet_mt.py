import math
from copy import deepcopy

import torch
from torch.nn import functional as F

from cpc.model.unet import BinaryUNet
from cpc.train_utils import calculate_train_steps


class BinaryUNetMT(BinaryUNet):
    args_schema = {
        **BinaryUNet.args_schema,
        "base_ema": (float, 0.996, "Exponential moving average parameter for the teacher network."),
        "unlabeled_weight": (float, 0.01, "Weightage given to the unlabeled term.")
    }

    def __init__(self, base_ema, unlabeled_weight, **kwargs):
        super().__init__(**kwargs)

        self.base_ema = base_ema
        self.unlabeled_weight = unlabeled_weight

        # teacher network
        self.teacher_network = deepcopy(self)
        self.teacher_network.requires_grad_(False)

        self.total_steps = ...

    def on_train_start(self):
        _, self.total_steps = calculate_train_steps(self)

    def training_step(self, batch, batch_idx):
        (labeled, masks), unlabeled = batch

        logits = self(torch.cat([labeled, unlabeled]))[-1].squeeze(1)
        pos_weight = torch.tensor([self.pos_weight]).to(self.device)

        labeled_logits = logits[:len(labeled)]
        labeled_seg_loss = F.binary_cross_entropy_with_logits(
            labeled_logits, masks.float(), pos_weight=pos_weight
        )
        self.log("train/labeled_seg_loss", labeled_seg_loss)

        unlabeled_logits = logits[len(labeled):]
        unlabeled_preds = torch.sigmoid(unlabeled_logits)
        with torch.no_grad():
            teacher_logits = self.teacher_network(unlabeled)[-1].squeeze(1)
            teacher_preds = torch.sigmoid(teacher_logits)
        unlabeled_const_loss = F.mse_loss(unlabeled_preds, teacher_preds)
        unlabeled_const_loss *= self.unlabeled_weight
        self.log("train/unlabeled_const_loss", unlabeled_const_loss)

        loss = labeled_seg_loss + unlabeled_const_loss
        self.log("train/loss", loss)

        return loss

    def _get_ema(self):
        current_step = self.global_step

        return 1 - (1 - self.base_ema) * (math.cos((current_step * math.pi) / self.total_steps) + 1) / 2

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        ema = self._get_ema()
        self.log("ema", ema, rank_zero_only=True)

        # update teacher network
        for student_param, teacher_param in zip(self.parameters(), self.teacher_network.parameters()):
            teacher_param.data = ema * teacher_param.data + (1 - ema) * student_param.data
