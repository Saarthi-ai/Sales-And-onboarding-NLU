from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


class AzureMLLogger(Logger):
    """PyTorch Lightning logger for AzureML training runs.

    Args:
        azureml_run: AzureML Run object corresponding to the training job.
    """
    def __init__(self, azureml_run):
        self.run = azureml_run
    
    @property
    def name(self):
        return 'azureml_pl_logger'
    
    @property
    def version(self):
        return "1.0.0"
    
    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            self.run.log(k, v, step)
    
    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass


class OptunaLogger(Logger):
    """PyTorch Lightning logger for Optuna hyperparameter search runs. Logs the last value of loss and validation loss.
    """    
    def __init__(self):
        self.cache = {
            'loss': [],
            'val_loss': []
        }

    @property
    def name(self):
        return 'optuna_pl_logger'
    
    @property
    def version(self):
        return "1.0.0"
    
    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            if k in ('loss', 'val_loss'):
                self.cache[k].append(v)
    
    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass
