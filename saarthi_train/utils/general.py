import torch


def get_teacher_checkpoint(model_name):
    """Utility function to get Hugging Face checkpoint from transformer model name.

    Args:
        model_name (str): Name of the transformer model.

    Returns:
        str: Hugging Face checkpoint of corresponding model name.
    """    
    checkpoints = {
        'bert': 'bert-base-uncased',
        'xlm-roberta': 'xlm-roberta-base',
        'muril': 'google/muril-base-cased'
    }
    return checkpoints[model_name]


def get_dtype_mapping_for_quantization(torch_type):
    """Utility function to map torch data type to torch quantization data type.

    Args:
        torch_type (str): Data type of the tensor.

    Returns:
        dtype: PyTorch datatype corresponding to the input string.
    """    
    type_map = {
        'int8': torch.qint8,
        'float16': torch.float16
    }

    return type_map[torch_type]
