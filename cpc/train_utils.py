import torch


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def calculate_train_steps(pl_module):
    trainer = pl_module.trainer
    num_devices = max(1, trainer.num_gpus, trainer.num_processes)
    if trainer.tpu_cores:
        num_devices = max(num_devices, trainer.tpu_cores)

    steps_per_epoch = (len(pl_module.train_dataloader()) // num_devices) // trainer.accumulate_grad_batches
    total_steps = steps_per_epoch * trainer.max_epochs
    if trainer.max_steps is not None:
        total_steps = min(total_steps, trainer.max_steps)

    return steps_per_epoch, total_steps
