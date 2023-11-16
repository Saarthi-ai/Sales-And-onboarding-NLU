import os
import json
import torch
from transformers import AutoTokenizer

from ..data.tokenizers import Tokenizer
from ..models.backbones import LSTMBackbone
from ..utils import TEXT_CLASSIFICATION_TASK_NAMES
from ..utils.general import get_teacher_checkpoint
from ..utils.model import select_head, select_student_model, select_transformer_model


# TODO: Add support for non-production inference (i.e. without torchscript).
def get_finetuned_teacher(output_path):
    """Loads trained checkpoint for teacher model saved in TorchScript format.

    Args:
        output_path (str): Path where the trained TorchScript teacher model is saved.

    Returns:
        model: Imported model.
        tokenizer: Models' tokenizer
        max_seq_len: Maximum sequence length of the model.
        task_name: Name of the task that the model was trained on.
    """    
    with open(os.path.join(output_path, 'training_config.json'), 'r') as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(get_teacher_checkpoint(config['teacher_name']))

    model_path = os.path.join(output_path, 'teacher', 'lightning_logs', 'version_0', 'checkpoints', 'model.pt')
    model = torch.jit.load(model_path)

    return model, tokenizer, config['max_seq_len'], config['task']


def build_teacher_from_checkpoint(output_path):
    """Builds a teacher model from a saved training checkpoint and returns it.

    Args:
        output_path (str): Path where the trained teacher model is saved.

    Returns:
        model: Trained teacher model
    """    
    with open(os.path.join(output_path, 'training_config.json'), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(output_path, 'labels.json'), 'r') as f:
        label_map = json.load(f)
    
    
    model = select_transformer_model(config['task']).from_pretrained(config['teacher_name'], config['task'], label_map, 10, 1000, None)

    checkpoint = torch.load(os.path.join(output_path, 'teacher', 'lightning_logs', 'version_0', 'checkpoints', 'teacher_checkpoint.ckpt'), map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    return model


def get_finetuned_student(output_path):
    """Loads trained checkpoint for student model saved in TorchScript format.

    Args:
        output_path (str): Path where the trained TorchScript student model is saved.

    Returns:
        model: Imported model.
        tokenizer: Models' tokenizer
        max_seq_len: Maximum sequence length of the model.
        task_name: Name of the task that the model was trained on.
    """    
    with open(os.path.join(output_path, 'training_config.json'), 'r') as f:
        config = json.load(f)

    with open(os.path.join(output_path, 'student', 'distil_vocab.json'), 'r') as f:
        distil_vocab = json.load(f)
    
    teacher_ckpt = get_teacher_checkpoint(config['teacher_name'])
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_ckpt)
    distil_tokenizer = Tokenizer(teacher_tokenizer)
    distil_tokenizer.load_vocabulary(distil_vocab)
    
    model_path = os.path.join(output_path, 'student', 'lightning_logs', 'version_0', 'checkpoints', 'model.pt')
    model = torch.jit.load(model_path)

    return model, distil_tokenizer, config['max_seq_len'], config['task']


# TODO: Fix proj_dim hardcoding
def build_student_from_checkpoint(output_path):
    """Builds a teacher model from a saved training checkpoint and returns it.

    Args:
        output_path (str): Path where the trained student model is saved.

    Returns:
        model: Trained student model
    """    
    with open(os.path.join(output_path, 'labels.json'), 'r') as f:
        label_map = json.load(f)

    with open(os.path.join(output_path, 'training_config.json'), 'r') as f:
        config = json.load(f)

    with open(os.path.join(output_path, 'student', 'distil_vocab.json'), 'r') as f:
        distil_vocab = json.load(f)

    teacher_ckpt = get_teacher_checkpoint(config['teacher_name'])
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_ckpt)
    distil_tokenizer = Tokenizer(teacher_tokenizer)
    distil_tokenizer.load_vocabulary(distil_vocab)

    checkpoint_path = os.path.join(output_path, 'student', 'lightning_logs', 'version_0', 'checkpoints', 'last.ckpt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    backbone = LSTMBackbone(
        lstm_dim=config['student_lstm_dim'],
        lstm_layers=config['student_lstm_layers'],
        lstm_dropout=0,
        emb_dim=config['student_word_emb_dim'],
        distil_vocab=distil_vocab,
        pt_vocab=teacher_tokenizer.get_vocab(),
        pt_embeddings=None,
        padding_idx=distil_tokenizer.get_pad_index(),
        bidirectional=True,
        proj_dim=768,
        get_pooled_output=True if config['task'] in TEXT_CLASSIFICATION_TASK_NAMES else False
    )
    classifier_head = select_head(config['task'])(backbone._get_output_dim(), config['task'], label_map)

    model = select_student_model(config['task'])(backbone, classifier_head, label_map, None)
    model.load_state_dict(checkpoint['state_dict'])

    return model
