import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning.pytorch as pl

from .backbones import TransformerBackbone
from .layers import SequenceClassificationOutputHead

from ..utils.losses import ClassBalancedFocalLossForSequences
from ..utils.optim_strategies import get_optimizers_for_transformer_finetuning


class TransformerSequenceClassifier(pl.LightningModule):
    """Sequence classification lightning model with a transformer backbone.

    Args:
        backbone: Transformer backbone.
        classification_head: Sequence classification head.
        max_epochs (int): Maximum number of training epochs.
        total_batches (int): Total number of training batches.
        samples_per_class (list): Number of samples per label class.
    """    
    def __init__(self, backbone, classification_head, max_epochs, total_batches, samples_per_class):
        super().__init__()
        self.max_epochs = max_epochs
        self.total_batches = total_batches

        self.backbone = backbone
        self.classification_head = classification_head

        self.model_name = self.backbone.model_name
        self.task_name = self.classification_head.task_name
        self.label_map = self.classification_head.get_label_map()
        self.loss = ClassBalancedFocalLossForSequences(samples_per_class=samples_per_class)

        self.save_hyperparameters(ignore=['backbone', 'classification_head'])

    @staticmethod
    def from_pretrained(model_name, task_name, label_map, max_epochs, total_batches, samples_per_class, dropout=None):
        """Initialize a pre-trained model using checkpoints from huggingface.

        Args:
            model_name (str): Name of the transformer to be used in the backbone. Currently supported: 'xlm-roberta', 'bert', 'muril'.
            task_name (str): Name of the sequence classification task. For ex - ner, pos, etc.
            label_map (dict): Dictionary containing lists of all the output labels.
            max_epochs (int): Maximum number of training epochs.
            total_batches (int): Total number of training batches.
            samples_per_class (list): Number of samples per label class.
            dropout (float, optional): Dropout probability. Defaults to None.

        Returns:
            TransformerSequenceClassifier: Sequence classification lightning model.
        """        
        backbone = TransformerBackbone(model_name, 'sequence_classification', dropout)
        backbone_output_dim = backbone._get_output_size()
        classification_head = SequenceClassificationOutputHead(backbone_output_dim, task_name, label_map)
        
        return TransformerSequenceClassifier(backbone, classification_head, max_epochs, total_batches, samples_per_class)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        features = self.backbone(input_ids, attention_mask, token_type_ids)
        return self.classification_head(features)

    def get_intermediate_layer_output(self, x, layer_number):
        with torch.no_grad():
            out = self.backbone.get_intermediate_output(x, layer_number)

        return out

    def training_step(self, batch, batch_idx):
        out = self.forward(**batch['text_token_ids'])

        loss = self.loss(out, batch[self.task_name])
        self.log('loss', loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(**batch['text_token_ids'])

        loss = self.loss(out, batch[self.task_name])
        self.log('val_loss', loss.item(), sync_dist=True)

    def configure_optimizers(self):
        return get_optimizers_for_transformer_finetuning(self.model_name, self.parameters(), self.total_batches)

    def get_preds(self, model_out):
        probs = F.softmax(model_out, dim=-1)
        return torch.argmax(probs, dim=-1)

    def freeze_backbone(self):
        self.backbone.requires_grad_(requires_grad=False)

    def freeze_model(self):
        self.requires_grad_(requires_grad=False)

    def unfreeze_backbone(self):
        self.backbone.requires_grad_(requires_grad=True)

    def unfreeze_model(self):
        self.requires_grad_(requires_grad=True)

    def get_teacher_checkpoint(self):
        return self.backbone._get_pretrained_checkpoint_name()


class DistilledSequenceClassifier(pl.LightningModule):
    """Sequence classification lightning model with a student backbone.

    Args:
        backbone: Student backbone.
        classification_head: Sequence classification head.
        label_map (dict): Dictionary containing lists of all the output labels.
        samples_per_class (list): Number of samples per label class.
    """    
    def __init__(self, backbone, classification_head, label_map, samples_per_class):
        super().__init__()
        self.label_map = label_map

        self.backbone = backbone
        self.classification_head = classification_head

        if samples_per_class:
            self.loss = ClassBalancedFocalLossForSequences(samples_per_class=samples_per_class)
            self.task_name = self.classification_head.task_name
            self.backbone_unfreezing_pointer = 0
            self.backbone_layers = [(name, module) for name, module in self.backbone.named_children()]
            self.backbone_layers.reverse()

        self.save_hyperparameters(ignore=['backbone', 'classification_head'])

    def forward(self, input_ids):
        features = self.backbone(input_ids)
        return self.classification_head(features)
    
    @torch.jit.ignore
    def training_step(self, batch, batch_idx):
        out = self.forward(batch['text_token_ids']['input_ids'])

        loss = self.loss(out, batch[self.task_name])
        self.log('loss', loss.item(), prog_bar=True, sync_dist=True)

        return loss
    
    @torch.jit.ignore
    def validation_step(self, batch, batch_idx):
        out = self.forward(batch['text_token_ids']['input_ids'])
        loss = self.loss(out, batch[self.task_name])
        self.log('val_loss', loss, sync_dist=True)

    @torch.jit.ignore
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        # Set `T_max` to 51 after looking at BFSI training runs.
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=51, eta_min=1e-8)

        return [optimizer], [lr_scheduler]
    
    @torch.jit.ignore
    def get_preds(self, model_out):
        probs = F.softmax(model_out, dim=-1)
        return torch.argmax(probs, dim=-1)

    @torch.jit.ignore
    def unfreeze_next_backbone_layer(self):
        print(f'Unfreezing layer: {self.backbone_layers[self.backbone_unfreezing_pointer][0]}')
        self.backbone_layers[self.backbone_unfreezing_pointer][1].requires_grad_(True)
        self.backbone_unfreezing_pointer = self.backbone_unfreezing_pointer + 1
    
    @torch.jit.ignore
    def get_number_of_required_distillation_epochs(self):
        return len(self.backbone_layers)


class TorchScriptXLMRSequenceClassifier(pl.LightningModule):
    """TorchScript-able class for XLM RoBERTa sequence classifier.

    Args:
        ts_backbone: Scripted/Traced XLMR backbone.
        ts_classification_head: Scripted/Traced classification head.
    """
    def __init__(self, ts_backbone, ts_classification_head):
        super().__init__()

        self.backbone = ts_backbone
        self.classification_head = ts_classification_head
        self.label_map = self.classification_head.get_label_map()
    
    def forward(self, input_ids, attention_mask):
        features = self.backbone(input_ids, attention_mask)
        return self.classification_head(features)


class TorchScriptBERTSequenceClassifier(pl.LightningModule):
    """TorchScript-able class for BERT sequence classifier.

    Args:
        ts_backbone: Scripted/Traced BERT backbone.
        ts_classification_head: Scripted/Traced classification head.
    """    
    def __init__(self, ts_backbone, ts_classification_head):
        super().__init__()

        self.backbone = ts_backbone
        self.classification_head = ts_classification_head
        self.label_map = self.classification_head.get_label_map()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        features = self.backbone(input_ids, attention_mask, token_type_ids)
        return self.classification_head(features)
