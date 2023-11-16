import torchmetrics
import torch.nn as nn


def get_multiclass_accuracy(label_map):
    """Getter function for accuracy in a multi-output text classification scenario.

    Args:
        label_map (dict): Dictionary containing lists of all the output labels.

    Returns:
        torch.nn.ModuleDict: ModuleDict containing accuracy for each model output.
    """    
    return nn.ModuleDict({
        out_name: torchmetrics.Accuracy(task='multiclass', num_classes=len(out_values))
        for out_name, out_values in label_map.items()
    })