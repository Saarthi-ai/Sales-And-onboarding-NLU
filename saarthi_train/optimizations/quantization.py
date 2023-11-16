import torch
from torch.ao.quantization import quantize_dynamic
from ..utils.general import get_dtype_mapping_for_quantization


def dynamic_quantization_for_transformers(model, dtype='int8'):
    """Dynamic quantization helper function for Hugging Face transformer models.

    Args:
        model: Transformer model to be quantized.
        dtype (str, optional): Datatype to be quantized to. Defaults to 'int8'.

    Returns:
        model: Quantized model
    """
    return quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=get_dtype_mapping_for_quantization(dtype))


def dynamic_quantization_for_lstms(model, dtype='int8'):
    """Dynamic quantization helper function for LSTM models.

    Args:
        model: LSTM model to be quantized.
        dtype (str, optional): Datatype to be quantized to. Defaults to 'int8'.

    Returns:
        model: Quantized model
    """
    return quantize_dynamic(model, qconfig_spec={torch.nn.LSTM, torch.nn.Linear}, dtype=get_dtype_mapping_for_quantization(dtype))
