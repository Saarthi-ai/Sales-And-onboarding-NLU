from torch.optim import AdamW
from .schedulers import LinearWithWarmupSchedule


def get_optimizers_for_transformer_finetuning(model_name, params, total_batches):
    """Getter method for optimizers for transformer finetuning. Based on "On the Stability of Fine-Tuning BERT".

    Args:
        model_name (str): Name of the transformer model.
        params: Model parameters.
        total_batches (int): Total number of training batches.

    Returns:
        list: List of optimizers for the transformer model.
    """    
    if 'roberta' in model_name:
        return optimizers_for_roberta(params, total_batches)
    if 'bert' in model_name or 'muril' in model_name:
        return optimizers_for_bert(params, total_batches)


def optimizers_for_roberta(params, total_batches):
    opt = AdamW(params, lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
    lrs = LinearWithWarmupSchedule(opt,
                                   warmup_ratio=0.1,
                                   total_batches=total_batches,
                                   start_lr=1e-5,
                                   end_lr=3e-5)
    
    return ([opt], [{'scheduler': lrs, 'interval': 'step'}])


def optimizers_for_bert(params, total_batches):
    opt = AdamW(params, lr=1e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    lrs = LinearWithWarmupSchedule(opt,
                                   warmup_ratio=0.1,
                                   total_batches=total_batches,
                                   start_lr=1e-5,
                                   end_lr=5e-5)
    
    return ([opt], [{'scheduler': lrs, 'interval': 'step'}])
