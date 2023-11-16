import torch
import torch.nn as nn
from balanced_loss import Loss


def get_multi_output_class_balanced_focal_loss(label_dict, samples_per_class):
    """Utility function that returns class balanced focal loss for multi-output text classification scenarios.

    Args:
        label_dict (dict): Dictionary containing lists of all the output labels.
        samples_per_class (list): Samples per class per output.

    Returns:
        dict: Dictionary containing class balanced focal loss for each output.
    """    
    return {
        classname: get_class_balanced_focal_loss(samples_per_class[classname]) 
        for classname in label_dict.keys()
    }


def get_class_balanced_focal_loss(samples_per_class):
    """Getter method for class balanced focal loss.

    Args:
        samples_per_class (list): Samples per class per output.

    Returns:
        Loss: Class balanced focal loss.
    """    
    return Loss(
        loss_type='focal_loss',
        samples_per_class=samples_per_class,
        class_balanced=True
    )


class NonPaddingTokenLoss(nn.Module):
    def __init__(self, loss_fn, pad_index):
        super(NonPaddingTokenLoss, self).__init__()
        self.loss_fn = loss_fn
        self.pad_idx = pad_index

    def forward(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)

        mask = (y_true != self.pad_idx).float()
        loss = loss * mask
        return torch.sum(loss) / torch.sum(mask)


class LossFunctionForSequences(nn.Module):
    """Base class of loss function for sequence classification tasks.

    Args:
        loss_fn (func): Specific loss function to be used.
        **kwargs: Keyword arguments to be passed to the loss function.
    """    
    def __init__(self, loss_fn, **kwargs):
        super(LossFunctionForSequences, self).__init__()
        self.loss_fn = loss_fn(**kwargs)
    
    def forward(self, y_preds, y_true):
        losses_per_sequence = [self.loss_fn(sequence, true_labels) for sequence, true_labels in zip(y_preds, y_true)]
        return torch.sum(torch.stack(losses_per_sequence))


class CrossEntropyLossForSequences(LossFunctionForSequences):
    """Cross entropy loss for sequence classification tasks.

    Args:
        reduction (str): 'mean' or 'sum' aggregation for the cross entropy loss function.
    """    
    def __init__(self, reduction):
        super().__init__(nn.CrossEntropyLoss, reduction=reduction)


class ClassBalancedFocalLossForSequences(LossFunctionForSequences):
    """Class balanced focal loss for sequence classification tasks.

    Args:
        samples_per_class (list): Number of samples per class.
    """    
    def __init__(self, samples_per_class):
        super().__init__(get_class_balanced_focal_loss, samples_per_class=samples_per_class)
