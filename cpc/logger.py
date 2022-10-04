from pytorch_lightning.loggers import TensorBoardLogger


def get_logger(root_dir):
    logger = TensorBoardLogger(
        save_dir=root_dir, name="lightning_logs", default_hp_metric=False
    )

    return logger
