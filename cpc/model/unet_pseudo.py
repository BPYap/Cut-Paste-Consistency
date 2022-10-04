import torch
from torch.nn import functional as F

from cpc.model.unet import BinaryUNet


class BinaryUNetPseudo(BinaryUNet):
    args_schema = {
        **BinaryUNet.args_schema,
        "pseudo_weight": (float, 0.5, "Weightage given to the pseudo-labeled term")
    }

    def __init__(self, pseudo_weight, **kwargs):
        super().__init__(**kwargs)

        self.pseudo_weight = pseudo_weight

    def training_step(self, batch, batch_idx):
        (labeled_images, labeled_masks), (unlabeled_images, pseudo_masks) = batch

        logits = self(torch.cat([labeled_images, unlabeled_images]))[-1].squeeze(1)
        pos_weight = torch.tensor([self.pos_weight]).to(self.device)

        labeled_logits = logits[:len(labeled_images)]
        labeled_loss = F.binary_cross_entropy_with_logits(labeled_logits, labeled_masks.float(), pos_weight=pos_weight)
        self.log("train/labeled_loss", labeled_loss)

        pseudo_logits = logits[len(labeled_images):]
        pseudo_loss = F.binary_cross_entropy_with_logits(pseudo_logits, pseudo_masks.float(), pos_weight=pos_weight)
        self.log("train/pseudo_loss", pseudo_loss)

        loss = labeled_loss + self.pseudo_weight * pseudo_loss
        self.log("train/loss", loss)

        return loss
