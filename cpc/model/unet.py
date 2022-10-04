import torch
import torchmetrics
from pl_bolts.models.vision import UNet as pl_UNet
from torch.nn import functional as F
from torchmetrics.functional import auc

from cpc.config import _list_or_int
from cpc.image_utils import crop_inference, resize_inference
from cpc.model.base_model import BaseModel


class BinaryUNet(BaseModel):
    args_schema = {
        **BaseModel.args_schema,
        "input_channels": (int, 3, "Number of channels in input images."),
        "num_layers": (int, 5, "Number of layers in each side of U-net."),
        "features_start": (int, 64, "Number of features in first layer."),
        "pos_weight": (float, 1.0, "Weight for positive samples."),
        "inference_mode": (str, "crop", "Inference mode. Choose from ['crop', 'resize']."),
        "inference_size": (_list_or_int, "256", "Size information during inference."
                                                "If `inference_mode == 'crop'`, this is the sliding window size, "
                                                "if `inference_mode =='resize'`, this is the downscale size."),
        "aupr_in_cpu": (bool, False, "Compute area under precision-recall curve in CPU instead of GPU."),
        "disable_aupr": (bool, False, "Disable the AUC-PR metrics.")
    }

    def __init__(self, input_channels, num_layers, features_start, pos_weight,
                 inference_mode, inference_size, aupr_in_cpu, disable_aupr, **kwargs):
        super().__init__(**kwargs)
        hparams = {"num_layers": num_layers, "features_start": features_start, "pos_weight": pos_weight}
        self.save_hyperparameters(hparams)

        self.num_layers = num_layers
        self.features_start = features_start
        self.model = pl_UNet(
            num_classes=1,
            input_channels=input_channels,
            num_layers=num_layers,
            features_start=features_start,
            bilinear=False
        )

        self.pos_weight = pos_weight
        self.inference_mode = inference_mode
        self.inference_size = inference_size
        self.aupr_to_cpu = aupr_in_cpu
        self.disable_aupr = disable_aupr
        self.metrics = torch.nn.ModuleDict()
        for prefix in ["val", "test"]:
            self.metrics[f"{prefix}_f1"] = torchmetrics.F1(threshold=0.5, average='micro')
            self.metrics[f"{prefix}_jaccard"] = torchmetrics.IoU(num_classes=2, threshold=0.5, reduction='none')
            self.metrics[f"{prefix}_pr"] = torchmetrics.PrecisionRecallCurve()

    def _forward_encoder(self, x):
        model = self.model
        features = [model.layers[0](x)]
        for layer in model.layers[1: model.num_layers]:
            features.append(layer(features[-1]))

        return features

    def _forward_decoder(self, features):
        model = self.model
        temp = features[-1]
        for i, layer in enumerate(model.layers[model.num_layers: -1]):
            temp = layer(temp, features[-2 - i])

        return model.layers[-1](temp)

    def forward(self, x):
        features = self._forward_encoder(x)
        logits = self._forward_decoder(features)

        return features, logits

    def on_train_start(self):
        self.register_log_metrics({"val/f1": 0, "val/aupr": 0})

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)[-1].squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=torch.tensor([self.pos_weight]).to(self.device))

        self.log("train/loss", loss)

        return loss

    def _shared_step(self, batch, prefix):
        x, y = batch
        inference_function = crop_inference if self.inference_mode == 'crop' else resize_inference
        y_hat = inference_function(self, x, self.inference_size)
        self.metrics[f"{prefix}_f1"].update(y_hat, y)
        self.metrics[f"{prefix}_jaccard"].update(y_hat, y)
        if not self.disable_aupr:
            if self.aupr_to_cpu:
                self.metrics[f"{prefix}_pr"].update(y_hat.cpu(), y.cpu())
            else:
                self.metrics[f"{prefix}_pr"].update(y_hat, y)

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_train_epoch_end(self, unused=None):
        pass

    def _shared_on_epoch_end(self, prefix):
        f1 = self.metrics[f"{prefix}_f1"].compute()
        self.metrics[f"{prefix}_f1"].reset()
        self.log(f"{prefix}/f1", f1, rank_zero_only=True)

        jaccard = self.metrics[f"{prefix}_jaccard"].compute()[1]
        self.metrics[f"{prefix}_jaccard"].reset()
        self.log(f"{prefix}/jaccard", jaccard, rank_zero_only=True)

        if not self.disable_aupr:
            p, r, _ = self.metrics[f"{prefix}_pr"].compute()
            self.metrics[f"{prefix}_pr"].reset()
            aupr = auc(r, p)
            self.log(f"{prefix}/aupr", aupr, rank_zero_only=True)

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end("val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end("test")
