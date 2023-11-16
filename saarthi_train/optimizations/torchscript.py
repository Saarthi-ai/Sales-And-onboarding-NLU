import torch
from transformers import AutoTokenizer
from ..models.text_classifiers import TorchScriptXLMRTextClassifier, TorchScriptBERTTextClassifier


def convert_lightning_transformer_classifier_to_ts(model, training_config, output_dir):
    """Helper function to convert lightning transformer classifier to TorchScript.

    Args:
        model (LightningModule): Transformer classifier to be converted to TorchScript.
        training_config (dict): Training configuration.
        output_dir (str): Path to save the serialized model to.
    """    
    tokenizer = AutoTokenizer.from_pretrained(model.get_teacher_checkpoint())
    example_input = dict(tokenizer('hello', padding='max_length', max_length=training_config['max_seq_len'], return_tensors='pt'))

    backbone = model.backbone
    head = model.classification_head
    backbone_output_size = backbone._get_output_size()

    traced_backbone = torch.jit.trace(backbone, example_kwarg_inputs=example_input)
    traced_head = torch.jit.trace(head, example_inputs=torch.randn(1, backbone_output_size), strict=False)

    if model.model_name in ['xlm-roberta']:
        scriptable_model = TorchScriptXLMRTextClassifier(traced_backbone, traced_head)
    elif model.model_name in ['bert', 'muril']:
        scriptable_model = TorchScriptBERTTextClassifier(traced_backbone, traced_head)

    scriptable_model.to_torchscript(file_path=output_dir)


def convert_lightning_lstm_classifier_to_ts(model, output_dir):
    """Helper function to convert lightning LSTM text classifier to TorchScript.

    Args:
        model (LightningModule): LSTM classifier to be converted to TorchScript.
        output_dir (str): Path to save the serialized model to.
    """    
    model.to_torchscript(file_path=output_dir)
