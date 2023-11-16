import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from  .layers import DistilledEmbeddingLayer
from ..utils.general import get_teacher_checkpoint


class TransformerBackbone(nn.Module):
    """Transformer based feature extraction backbone for text modality.

    Args:
        model_name (str): Name of the transformer model to use. Currently supported: 'xlm-roberta', 'bert', 'muril'.
        task (str): Name of the training task.
        dropout (float, optional): Dropout probability. Defaults to None.
    """    
    def __init__(self, model_name, task, dropout=None):
        super().__init__()
        self.model_name = model_name
        self.checkpoint = get_teacher_checkpoint(self.model_name)
        config = AutoConfig.from_pretrained(self.checkpoint, output_hidden_states=True)
        self.model_output = self._select_model_output(task)
        self.model_output_size = config.hidden_size

        if dropout:
            config.attention_probs_dropout_prob = dropout
            config.hidden_dropout_prob = dropout

        self.model = AutoModel.from_pretrained(self.checkpoint, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if self.model_name in ['xlm-roberta']:
            features = self.model(input_ids=input_ids, attention_mask=attention_mask)[self.model_output]
        if self.model_name in ['bert', 'muril']:
            assert token_type_ids is not None
            features = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )[self.model_output]

        return features

    @torch.jit.ignore
    def get_intermediate_output(self, x, layer_number):
        return self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'])['hidden_states'][layer_number]

    @torch.jit.ignore
    def _get_output_size(self):
        return self.model_output_size

    @torch.jit.ignore
    def _get_intermediate_layer_output_size(self, layer_number):
        return self.model.state_dict()[f'encoder.layer.{layer_number}.output.dense.weight'].size()[0]

    @torch.jit.ignore
    def _get_embedding_weights(self):
        return self.model.embeddings.word_embeddings.weight.data.numpy()

    @torch.jit.ignore
    def _get_pretrained_checkpoint_name(self):
        return self.checkpoint
    
    @torch.jit.ignore
    def _select_model_output(self, task):
        if task == 'text_classification':
            return 'pooler_output'
        elif task == 'sequence_classification':
            return 'last_hidden_state'


class LSTMBackbone(nn.Module):
    """LSTM based feature extraction backbone for Xtremedistil student model text modality.

    Args:
        lstm_dim (int): Hidden dimension of LSTM layer.
        lstm_layers (int): Number of LSTM layers to use.
        lstm_dropout (float): Dropout between the LSTM layers.
        emb_dim (int): Embedding layer vector dimension.
        distil_vocab (dict): Vocabulary of the student model.
        pt_vocab (dict): Vocabulary of the teacher model.
        pt_embeddings (numpy.array): Teacher word embeddings.
        padding_idx (int): Index of the padding token.
        bidirectional (bool, optional): Whether the LSTM needs to be bidirectional. Defaults to False.
        proj_dim (int, optional): Dimension to project LSTM output to. Defaults to None.
        get_pooled_output (bool, optional): Whether to get the pooled output or not (for text classification tasks). Defaults to False.
    """    
    def __init__(self, lstm_dim, lstm_layers, lstm_dropout, emb_dim, distil_vocab, pt_vocab, pt_embeddings, padding_idx, bidirectional=False, proj_dim=None, get_pooled_output=False):
        super().__init__()

        print('Initializing distil embedding layer.')
        self.embedding_layer = DistilledEmbeddingLayer(
            distil_dim=emb_dim,
            distil_vocab=distil_vocab,
            pretrained_vocab=pt_vocab,
            pretrained_embeddings=pt_embeddings,
            padding_idx=padding_idx,
            freeze=False
        )
        print('Done')

        self.lstm_dim = lstm_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=self.lstm_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_out_dim = lstm_dim * 2 if bidirectional else lstm_dim
        self.proj_dim = proj_dim

        if self.proj_dim:
            self.projector = nn.Linear(lstm_out_dim, self.proj_dim)
        
        self.get_pooled_output = get_pooled_output

    def forward(self, input_ids):
        embedding = self.embedding_layer(input_ids).float()
        out, _ = self.lstm(embedding)

        if self.proj_dim:
            out = F.gelu(self.projector(out))

        if self.get_pooled_output:
            out = out[:, -1, :]

        # TODO: Experiment using this slicing as it contains context
        # from both forward and backward looks of the bidirectional LSTM.
        # if self.bidirectional:
        #     out = torch.cat([out[-1, :, :self.lstm_dim], out[0, :, self.lstm_dim:]], dim=-1)
        # else:
        #     out = out[-1]

        return out

    def _get_output_dim(self):
        dim = None
        if self.proj_dim:
            dim = self.proj_dim
        else:
            dim = self.lstm_dim * 2 if self.bidirectional else self.lstm_dim

        return dim
