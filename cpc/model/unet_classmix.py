import math
import random
from copy import deepcopy

import torch
from torch.nn import functional as F

from cpc.model.unet import BinaryUNet
from cpc.train_utils import calculate_train_steps


class BinaryUNetClassMix(BinaryUNet):
    args_schema = {
        **BinaryUNet.args_schema,
        "mean_teacher": (bool, False, "Use mean teacher model for pseudo-labelling."),
        "base_ema": (float, 0.996, "Exponential moving average parameter for the teacher network."),
        "unlabeled_weight": (float, 0.01, "Weightage given to the unlabeled term.")
    }

    def __init__(self, mean_teacher, base_ema, unlabeled_weight, **kwargs):
        super().__init__(**kwargs)

        self.base_ema = base_ema
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
    def _sample_indices(count):
        sampled_indices = []
        indices = set(range(count))
        for i in range(count):
            sampled_indices.append(random.choice(list(indices.difference({i}))))

        return sampled_indices

    def training_step(self, batch, batch_idx):
        (labeled, masks), unlabeled = batch

        # generate synthetic samples with ClassMix
        with torch.no_grad():
            pseudo_labeler = self.teacher_network if self.teacher_network is not None else self
            unlabeled_logits = pseudo_labeler(unlabeled)[-1].squeeze(1)
            unlabeled_masks = torch.sigmoid(unlabeled_logits) >= 0.5
            pair_indices = self._sample_indices(len(unlabeled))
            temp = ~unlabeled_masks
            synth_images = unlabeled_masks.unsqueeze(1) * unlabeled + temp.unsqueeze(1) * unlabeled[pair_indices]
            synth_logits = unlabeled_masks * unlabeled_logits + temp * unlabeled_logits[pair_indices]
            synth_masks = torch.sigmoid(synth_logits) >= 0.5

        pos_weight = torch.tensor([self.pos_weight]).to(self.device)
        logits = self(torch.cat([labeled, synth_images]))[-1].squeeze(1)

        labeled_logits = logits[:len(labeled)]
        labeled_seg_loss = F.binary_cross_entropy_with_logits(
            labeled_logits, masks.float(), pos_weight=pos_weight
        )
        self.log("train/labeled_seg_loss", labeled_seg_loss)

        synth_logits = logits[len(labeled):]
        unlabeled_seg_loss = F.binary_cross_entropy_with_logits(
            synth_logits, synth_masks.float(), pos_weight=pos_weight
        )
        unlabeled_seg_loss *= self.unlabeled_weight
        self.log("train/unlabeled_seg_loss", unlabeled_seg_loss)

        loss = labeled_seg_loss + unlabeled_seg_loss
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
