import math
import warnings
from torch.optim.lr_scheduler import LRScheduler


class CustomLRScheduler(LRScheduler):
    """Custom learning rate schedule found in the Xtremedistil code.

    Args:
        optimizer (nn.Optimizer): Optimizer to be used for training.
        max_lr (float): Maximum value for learning rate.
        end_lr (float): Final value for learning rate.
        warmup_epochs (int): Number of warmup epochs.
        total_epochs (int): Number of total training epochs.
        last_epoch (int): _description_. Defaults to -1.
        verbose (bool): Whether to output additional information in stdout. Defaults to False.
    """    
    def __init__(self, optimizer, max_lr=5e-5, end_lr=1e-7, warmup_epochs=10, total_epochs=100, last_epoch=-1, verbose=False):
        self.max_lr = max_lr
        self.end_lr = end_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch < self.warmup_epochs:
            lr = (self.max_lr / self.warmup_epochs) * (self.last_epoch + 1)
        else:
            lr = self.max_lr * math.exp(math.log(self.end_lr / self.max_lr) * (self.last_epoch - self.warmup_epochs + 1) / (self.total_epochs - self.warmup_epochs + 1))
        
        lr = float(lr)
        return [lr for _ in self.optimizer.param_groups]


class LinearWithWarmupSchedule(LRScheduler):
    """Linear learning rate schedule with linear increase during the warmup period.

    Args:
        optimizer (torch.nn.Optimizer): Optimizer to be used during training.
        warmup_ratio (float): Ratio of batches to be used for warmup.
        total_batches (int): Total number of training batches.
        start_lr (float): Starting value for learning rate.
        end_lr (float): Final value for learning rate.
        last_epoch (int): _description_. Defaults to -1.
        verbose (bool): Whether to output additional information in stdout. Defaults to False.
    """    
    def __init__(self, optimizer, warmup_ratio, total_batches, start_lr, end_lr, last_epoch=-1, verbose=False):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_batches = int(total_batches * warmup_ratio)
        self.delta = (self.end_lr - self.start_lr) / self.warmup_batches
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def get_lr(self):
        current_batch = self._step_count
        if current_batch <= self.warmup_batches:
            return [self.start_lr + self.delta * current_batch for _ in self.base_lrs]
        else:
            return [self.end_lr for _ in self.base_lrs]
