import torch
import numpy as np
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA

from ..utils import SEQUENCE_CLASSIFICATION_TASK_NAMES


class AutoClassificationOutputHead(nn.Module):
    """Classification head for text classification tasks that automatically figures out the number of output heads and their output dimensions based on the label dictionary.

    Args:
        input_dim (int): Input dimension.
        task_name (str): Name of the training task.
        label_map (dict): Dictionary containing lists of all the output labels.
    """    
    def __init__(self, input_dim, task_name, label_map):
        # Including task_name argument for consistency across API.
        super().__init__()
        self.label_map = label_map
        self.task_name = task_name
        self.heads = nn.ModuleDict(
            {
                classname: nn.Linear(input_dim, len(classvalues))
                for classname, classvalues in label_map.items()
                if classname not in SEQUENCE_CLASSIFICATION_TASK_NAMES
            }
        )

    def forward(self, x):
        return {classname: layer(x) for classname, layer in self.heads.items()}

    @torch.jit.ignore
    def get_keys(self):
        return self.heads.keys()

    @torch.jit.export
    def get_label_map(self):
        return self.label_map


class SequenceClassificationOutputHead(nn.Module):
    """Final output layer for sequence classification tasks. Determines the output dimension automatically by reading from the label dictionary.

    Args:
        input_dim (int): Input dimension.
        task_name (str): Name of the training task.
        label_map (dict): Dictionary containing lists of all the output labels.
    """    
    def __init__(self, input_dim, task_name, label_map):
        super().__init__()
        self.label_map = label_map
        self.task_name = task_name
        self.head = nn.Linear(input_dim, len(label_map[self.task_name]))

    def forward(self, x):
        return self.head(x)

    @torch.jit.export
    def get_label_map(self):
        return self.label_map


class DistilledEmbeddingLayer(nn.Module):
    """Embedding layer for the student model in Xtremedistil pipeline. Takes as input pretrained teacher model word embeddings and applies dimensionality reduction on them to match the student model's word embedding dimension. Alternatively, if no embeddings are provided, will randomly initialize the student embeddings instead.

    Args:
        distil_dim (int): Dimensions for student model embeddings.
        distil_vocab (dict): Vocabulary of student model.
        pretrained_vocab (dict): Vocabulary of teacher model.
        padding_idx (int): Index of padding token for student model.
        pretrained_embeddings (numpy.array, optional): Pretrained word embeddings of teacher model. Defaults to None.
        freeze (bool, optional): Whether to freeze the distilled embedding layer or not. Defaults to False.
        pca_batch_size (int, optional): Batch size for incremental PCA. Defaults to 1000.
    """    
    def __init__(self, distil_dim, distil_vocab, pretrained_vocab, padding_idx, pretrained_embeddings=None, freeze=False, pca_batch_size=1000):
        super().__init__()

        if pretrained_embeddings is not None:
            pca = IncrementalPCA(n_components=distil_dim, batch_size=pca_batch_size)
            downscaled_embeddings = pca.fit_transform(pretrained_embeddings)
            downscaled_mapping = {k:v for k,v in zip(pretrained_vocab, downscaled_embeddings)}
            # Converting to numpy array first because converting a list of numpy arrays to a tensor is slower.
            distil_embedding_weights = torch.tensor(np.array([downscaled_mapping[word] if (word in downscaled_mapping) else np.random.uniform(-0.1, 0.1, distil_dim) for word in distil_vocab.keys()]))
        else:
            distil_embedding_weights = torch.rand(len(distil_vocab), distil_dim)

        print('Initializing embedding layer')
        self.embedding = nn.Embedding.from_pretrained(
            distil_embedding_weights,
            freeze=freeze,
            padding_idx=padding_idx
        )

    def forward(self, x):
        return self.embedding(x)
