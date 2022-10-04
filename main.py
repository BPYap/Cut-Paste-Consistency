import argparse

from cpc.config import get_argument_parser, parse_and_process_arguments, get_trainer
from cpc.data import (
    IDRiDDataModule,
    IDRiDDataModuleSemi, IDRiDDataModuleExtra,
    IDRiDDataModuleBaseCP, IDRiDDataModulePseudoCP, IDRiDDataModuleCP,
    ICHDataModule,
    ICHDataModuleSemi, ICHDataModuleExtra,
    ICHDataModuleBaseCP, ICHDataModulePseudoCP, ICHDataModuleCP
)
from cpc.model import (
    BinaryUNet, BinaryUNetCP, BinaryUNetPseudo, BinaryUNetMT, BinaryUNetClassMix, BinaryUNetCutMix
)

DATA_MODULES = {
    "idrid": IDRiDDataModule,
    "idrid-semi": IDRiDDataModuleSemi,
    "idrid-st": IDRiDDataModuleExtra,
    "idrid-base-cp": IDRiDDataModuleBaseCP,
    "idrid-st-cp": IDRiDDataModulePseudoCP,
    "idrid-cp": IDRiDDataModuleCP,

    "ich": ICHDataModule,
    "ich-semi": ICHDataModuleSemi,
    "ich-st": ICHDataModuleExtra,
    "ich-base-cp": ICHDataModuleBaseCP,
    "ich-st-cp": ICHDataModulePseudoCP,
    "ich-cp": ICHDataModuleCP
}
LN_MODULES = {
    "unet": BinaryUNet,
    "unet-cp": BinaryUNetCP,
    "unet-pseudo": BinaryUNetPseudo,
    "unet-mt": BinaryUNetMT,
    "unet-classmix": BinaryUNetClassMix,
    "unet-cutmix": BinaryUNetCutMix
}
MODULE_COMPATIBILITY = {
    # input shape: (img, mask)
    "unet": {"idrid", "ich"},

    # input shape: (img, mask), (synth_img, synth_mask, background)
    "unet-cp": {"idrid-cp", "ich-cp"},

    # input shape: (img, mask), (synth_img, synth_mask)
    "unet-pseudo": {"idrid-st", "idrid-base-cp", "idrid-st-cp", "ich-st", "ich-base-cp", "ich-st-cp"},

    # input shape: (img, mask), img
    "unet-mt": {"idrid-semi", "ich-semi"},
    "unet-classmix": {"idrid-semi", "ich-semi"},
    "unet-cutmix": {"idrid-semi", "ich-semi"}
}


def check_compatibility(dm_name, model_name):
    if dm_name not in MODULE_COMPATIBILITY[model_name]:
        raise ValueError(f"'{model_name}' is not compatible with datamodule '{dm_name}'.")


def main(_args, _dm_cls, _model_cls):
    data_args, model_args, trainer_args, other_args = _args
    dm = _dm_cls(**data_args)
    model = _model_cls(**model_args)
    trainer = get_trainer(**trainer_args)

    if other_args["do_train"]:
        # Train
        if trainer_args["auto_lr_find"]:
            # run learning rate finder before training
            trainer.tune(model, dm)
        trainer.fit(model, dm)

    if other_args["do_test"]:
        # Test
        if not other_args["do_train"]:
            model = _model_cls.load_from_checkpoint(trainer_args["resume_from_checkpoint"], strict=False, **model_args)
        trainer.test(model, dm)


if __name__ == "__main__":
    entry_parser = argparse.ArgumentParser(add_help=False)
    entry_parser.add_argument('data_module', type=str, choices=list(DATA_MODULES.keys()))
    entry_parser.add_argument('model', type=str, choices=list(LN_MODULES.keys()))
    entry_args = entry_parser.parse_known_args()[0]

    check_compatibility(entry_args.data_module, entry_args.model)

    dm_cls = DATA_MODULES[entry_args.data_module]
    model_cls = LN_MODULES[entry_args.model]

    parser = get_argument_parser(dm_cls, model_cls, parents=[entry_parser])
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    args = parse_and_process_arguments(parser)

    main(args, dm_cls, model_cls)
