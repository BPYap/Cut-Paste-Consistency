import math
import warnings

import torch.optim as optim


class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    """Linearly warmup learning rate to a specified update steps then apply cosine annealing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of update steps in linear warmup.
        total_steps (int): Total number of update steps.
    """

    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        current_step = self.last_epoch
        if self.warmup_steps > 0 and current_step <= self.warmup_steps:
            factor = current_step / self.warmup_steps
        else:
            offset_steps = current_step - self.warmup_steps
            offset_total = self.total_steps - self.warmup_steps
            factor = (1 + math.cos(math.pi * offset_steps / offset_total)) / 2

        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupStepLR(optim.lr_scheduler._LRScheduler):
    """Linearly warmup learning rate to a specified update steps then apply learning rate decay at specified steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of update steps in linear warmup.
        decay_steps (list): List of steps to apply learning rate decay.
        factor (float): Factor of learning rate decay.
    """

    def __init__(self, optimizer, warmup_steps, decay_steps, factor):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.factor = factor
        self.cumulative_factor = 1

        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        current_step = self.last_epoch
        if self.warmup_steps > 0 and current_step <= self.warmup_steps:
            factor = current_step / self.warmup_steps
        else:
            if current_step in self.decay_steps:
                self.cumulative_factor *= self.factor

            factor = self.cumulative_factor

        return [base_lr * factor for base_lr in self.base_lrs]
