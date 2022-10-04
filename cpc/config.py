import argparse
import json
import os
from collections import defaultdict
from distutils.dir_util import copy_tree

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from cpc.logger import get_logger


def _list_or_int(s):
    if s[0] == '[' and s[-1] == ']':
        return [int(c) for c in s[1:-1].split(',')]
    else:
        return int(s)


def _float_or_int(s):
    if len(s.split('.')) == 2:
        return float(s)
    else:
        return int(s)


TRAINER_ARGS = {
    # name:                    (type, default, help)
    "default_root_dir":        (str, os.getcwd(), "Default directory for logs and weights."),
    "resume_from_checkpoint":  (str, None, "Path to model checkpoint (.ckpt)."),
    "gpus":                    (_list_or_int, 0, "Specify number of GPUs to use."),
    "num_nodes":               (int, 1, "Number of GPU nodes for distributed training."),
    "accelerator":             (str, None, "Choose from [None, 'ddp']."),
    "amp_backend":             (str, "native", "Backend for mixed precision. Choose from ['native', 'apex']."),
    "amp_level":               (str, "O2", "Optimization level to use for apex amp."),
    "precision":               (int, 32, "Choose from [16, 32, 64]."),
    "accumulate_grad_batches": (int, 1, "Number of steps to accumulate gradients."),
    "max_epochs":              (int, 1000, "Number of epochs to train for."),
    "max_steps":               (int, None, "Number of steps to train for."),
    "check_val_every_n_epoch": (int, 1, "Run validation loop for every n-th epochs."),
    "log_every_n_steps":       (int, 50, "Number of logging steps."),
    "num_sanity_val_steps":    (int, 0, "Run n batches of validation samples as sanity check before training."),
    "checkpoint_monitor":      (str, None, "Name of the quantity to determine the best checkpoint."),
    "early_stopping_patience": (int, -1, "Number of validation checks for early stopping based on the quantity "
                                         "specified in `checkpoint_monitor`."),
    "auto_lr_find":            (bool, False, "Automatically find a suitable learning rate before training.")
}

CODE_DIRS = ["cpc"]
PARSED_ARGS = None


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_args = defaultdict(set)

    def add_args(self, group_name, args_schema):
        group = self.add_argument_group(group_name)
        for name, param in args_schema.items():
            type_, default, help_ = param
            if type_ == bool:
                group.add_argument(
                    f"--{name}",
                    action='store_true',
                    help=help_
                )
            else:
                group.add_argument(
                    f"--{name}",
                    type=type_,
                    default=default,
                    help=help_,
                    metavar=default
                )
            self.group_args[group_name].add(name)

    def get_args(self, group_name):
        return self.group_args[group_name]


def get_argument_parser(dm_cls, model_cls, include_trainer_args=True, **kwargs):
    parser = _ArgumentParser(**kwargs)
    parser.add_args("data_args", dm_cls.args_schema)
    parser.add_args("model_args", model_cls.args_schema)
    if include_trainer_args:
        parser.add_args("trainer_args", TRAINER_ARGS)

    # other args
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility.",
        metavar=42
    )

    return parser


def parse_and_process_arguments(parser):
    args = parser.parse_args()
    data_args = dict()
    model_args = dict()
    trainer_args = dict()
    other_args = dict()
    for k, v in vars(args).items():
        if k in parser.get_args("data_args"):
            data_args[k] = v
        elif k in parser.get_args("model_args"):
            model_args[k] = v
        elif k in parser.get_args("trainer_args"):
            trainer_args[k] = v
        else:
            other_args[k] = v

    seed_everything(args.seed, workers=True)

    global PARSED_ARGS
    PARSED_ARGS = args

    return data_args, model_args, trainer_args, other_args


class _SaveArgsCallback(Callback):
    def on_init_end(self, _trainer):
        @rank_zero_only
        def save_args():
            log_dir = _trainer.logger.log_dir
            os.makedirs(log_dir)
            # save command line arguments
            with open(os.path.join(log_dir, "args.json"), 'w') as f:
                json.dump(vars(PARSED_ARGS), f, indent=4)

        save_args()


class _SaveSnapshotCallback(Callback):
    def on_train_start(self, _trainer, pl_module):
        @rank_zero_only
        def save_snapshots():
            log_dir = _trainer.logger.log_dir
            # save a copy of current codes
            for code_dir in CODE_DIRS:
                copy_tree(code_dir, os.path.join(log_dir, "snapshot", code_dir))

        save_snapshots()


def _get_callbacks(checkpoint_monitor, every_n_epochs, early_stopping_patience):
    callbacks = []

    # checkpoint callback
    if checkpoint_monitor is None:  # save the latest checkpoint
        callbacks.append(ModelCheckpoint())
    else:  # save the best checkpoint (evaluated on the monitored quantity)
        callbacks.append(
            ModelCheckpoint(
                filename="best_checkpoint",
                monitor=checkpoint_monitor,
                every_n_epochs=every_n_epochs,
                save_top_k=1,
                save_last=True,
                mode='max',
                verbose=True)
        )

    # save a copy of command line arguments
    callbacks.append(_SaveArgsCallback())

    # save a copy of codes when training start
    callbacks.append(_SaveSnapshotCallback())

    # monitor learning rate during the course of training
    callbacks.append(LearningRateMonitor(logging_interval='step', log_momentum=True))

    # stop training if there is no improvement in validation result
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_monitor,
                patience=early_stopping_patience,
                verbose=True,
                mode='max',
                strict=True,
                check_finite=True
            )
        )

    return callbacks


def get_trainer(**kwargs):
    # extract custom arguments
    checkpoint_monitor = kwargs["checkpoint_monitor"]
    del kwargs["checkpoint_monitor"]
    early_stopping_patience = kwargs["early_stopping_patience"]
    del kwargs["early_stopping_patience"]

    _kwargs = {
        "callbacks": _get_callbacks(checkpoint_monitor, kwargs["check_val_every_n_epoch"], early_stopping_patience),
        "checkpoint_callback": True,
        "log_gpu_memory": None,  # logging GPU memory might slow down training
        "logger": get_logger(kwargs["default_root_dir"]),
        "plugins": DDPPlugin(find_unused_parameters=False) if kwargs["accelerator"] == 'ddp' else None,
        "auto_lr_find": kwargs["auto_lr_find"]
    }
    _kwargs.update(kwargs)
    trainer = Trainer(**_kwargs)

    return trainer
