from cpc.data.hybrid_data import HybridDataModule
from cpc.data.idrid.idrid import IDRiDDataModule
from cpc.data.idrid.idrid_synth import IDRiDDataModuleSynth


class IDRiDDataModuleCP(HybridDataModule):
    args_schema = {
        **IDRiDDataModule.args_schema,
        "synth_split": (float, 0.4, "Proportion of synthetic samples in each mini-batch."),
        "num_synth": (int, 300, "Number of images to be synthesized from the unlabeled set."),
        "img_match": (bool, False, "Find matching foreground for each background using color hashing"),
        "mask_blur": (str, "none", "Apply blurring to the masks before pasting. "
                                   "Choose from ['none', 'gaussian', 'random']."),
        "background_blur": (str, "none", "Apply blurring to the background before pasting. "
                                         "Choose from ['none', 'gaussian', 'random'].")
    }

    def __init__(self, val_split, synth_split, num_synth, img_match, mask_blur, background_blur,
                 return_background=True, background_mask_dir=None, **kwargs):
        hparams = {
            "batch_size": kwargs["batch_size"],
            "synth_split": synth_split,
            "num_synth": num_synth,
            "img_match": img_match,
            "mask_blur": mask_blur,
            "background_blur": background_blur
        }
        self.save_hyperparameters(hparams)

        batch_size = kwargs["batch_size"]
        synth_batch_size = int(batch_size * synth_split)
        real_batch_size = batch_size - synth_batch_size
        del kwargs["batch_size"]
        self.dm = IDRiDDataModule(val_split, batch_size=real_batch_size, **kwargs)
        self.synth_dm = IDRiDDataModuleSynth(
            num_samples=num_synth, batch_size=synth_batch_size,
            img_match=img_match, mask_blur=mask_blur, background_blur=background_blur,
            return_background=return_background, background_mask_dir=background_mask_dir,
            **kwargs
        )

        super().__init__(self.dm, [self.synth_dm], merge_samples=False)


class IDRiDDataModuleBaseCP(IDRiDDataModuleCP):
    args_schema = {
        **IDRiDDataModuleCP.args_schema
    }

    def __init__(self, **kwargs):
        super().__init__(return_background=False, **kwargs)


class IDRiDDataModulePseudoCP(IDRiDDataModuleCP):
    args_schema = {
        **IDRiDDataModuleCP.args_schema,
        "background_mask_dir": (str, None, "Directory of background masks if available.")
    }

    def __init__(self, background_mask_dir, **kwargs):
        super().__init__(return_background=False, background_mask_dir=background_mask_dir, **kwargs)
        hparams = {
            "background_mask_dir": background_mask_dir
        }
        self.save_hyperparameters(hparams)
