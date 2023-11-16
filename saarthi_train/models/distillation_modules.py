import os
import copy
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from ..data.tokenizers import Tokenizer
from ..data.modules import RepresentationTransferDataModule

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from .backbones import LSTMBackbone
from ..utils.data import select_datamodule
from ..utils.model import select_head, \
                             select_student_model
from ..utils.metrics import get_multiclass_accuracy

from ..utils import TEXT_CLASSIFICATION_TASK_NAMES, \
                    SEQUENCE_CLASSIFICATION_TASK_NAMES


def run_xtremedistil(config, teacher_model, label_map, loggers):
    """Runs the Xtremedistil knowledge distillation pipeline.

    Args:
        config (dict): Training configuration.
        teacher_model (str): Name of the teacher model.
        label_map (dict): Dictionary containing lists of all the output labels.
        loggers (dict): Dictionary containing logger objects for each stage of training.

    Returns:
        model: Distilled student model.
    """    
    teacher_model.freeze_model()
    teacher_checkpoint = teacher_model.get_teacher_checkpoint()
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint)

    # Create distilled tokenizer
    distil_tokenizer = create_and_save_distilled_tokenizer(
        pt_tokenizer=teacher_tokenizer,
        data_path=config['input_path'],
        savedir=os.path.join(config['root_output_path'], config['student_output_folder'])
    )
    distil_vocab = distil_tokenizer.get_vocab()

    stage1model = XtremeDistilStage1Module(
        teacher=teacher_model,
        layer_number=config['teacher_layer_to_distil'],
        teacher_vocab=teacher_tokenizer.get_vocab(),
        lstm_dim=config['student_lstm_dim'],
        lstm_layers=config['student_lstm_layers'],
        lstm_dropout=config['student_lstm_dropout'],
        student_emb_dim=config['student_word_emb_dim'],
        student_vocab=distil_vocab,
        padding_idx=distil_tokenizer.get_pad_index()
    )

    representation_module = RepresentationTransferDataModule(
        path=config['input_path'],
        output_path=os.path.join(config['root_output_path'], config['student_output_folder']),
        label_map=label_map,
        max_seq_len=config['max_seq_len'],
        pt_tokenizer=teacher_tokenizer,
        distil_tokenizer=distil_tokenizer,
        batch_size=config['student_batch_size']
    )

    print('Initializing trainer')
    stage1trainer = pl.Trainer(
        max_epochs=config['student_epochs'],
        accelerator='gpu',
        devices=1,
        callbacks=[
            EarlyStopping(monitor='loss', patience=10),
            LearningRateMonitor(logging_interval='epoch')
        ],
        precision=config['precision'],
        enable_checkpointing=False,
        logger=loggers['student_stage_1'],
        fast_dev_run=config['fast_dev_run'],
    )
    print('Starting stage 1')
    stage1trainer.fit(stage1model, representation_module)

    stage1_student = stage1model.get_backbone()

    print('Initializing stage 2 distillation model')
    stage2model = XtremeDistilStage2Module(
        teacher=teacher_model,
        stage1_student=stage1_student,
        task=config['task'],
        dropout=0.2,
        label_map=label_map
    )

    print('Initialzing stage 2 trainer')
    stage2trainer = pl.Trainer(
        max_epochs=config['student_epochs'],
        accelerator='gpu',
        devices=1,
        callbacks=[
            EarlyStopping(monitor='loss', patience=10),
            LearningRateMonitor(logging_interval='epoch')
        ],
        precision=config['precision'],
        logger=loggers['student_stage_2'],
        enable_checkpointing=False,
        fast_dev_run=config['fast_dev_run'],
    )

    print('Starting stage 2')
    stage2trainer.fit(stage2model, representation_module)

    print('Initializing stage 3 model')
    stage3model = XtremeDistilStage3Module(stage2model, label_map)

    print('Starting stage 3')
    for _ in range(stage3model.get_number_of_required_distillation_epochs()):
        stage3trainer = pl.Trainer(
            max_epochs=config['student_epochs'],
            accelerator='gpu',
            devices=1,
            callbacks=[
                EarlyStopping(monitor='loss', patience=10),
                LearningRateMonitor(logging_interval='epoch')
            ],
            precision=config['precision'],
            logger=loggers['student_stage_3'],
            enable_checkpointing=False,
            fast_dev_run=config['fast_dev_run'],
        )
        stage3model.unfreeze_next_backbone_layer()
        stage3trainer.fit(stage3model, representation_module)

    DataModule = select_datamodule(config['task'])
    print('Initializing student stage 4 data module')
    student_datamodule = DataModule(
        path=config['input_path'],
        output_path=os.path.join(config['root_output_path'], config['student_output_folder']),
        task_name=config['task'],
        max_seq_len=config['max_seq_len'],
        batch_size=config['student_batch_size'],
        tokenizer=distil_tokenizer
    )
    # Dirty hack for making samples_per_class getter work.
    student_datamodule.prepare_data()
    student_datamodule.setup(stage='fit')
    # Dirty hack over. I'm sorry you had to see that.
    samples_per_class = student_datamodule.get_samples_per_class()

    print('Getting final student model fron stage 3 module')
    student_model = stage3model.get_final_model(label_map, samples_per_class)

    print('Initializing stage 4 trainer')
    stage4trainer = pl.Trainer(
        max_epochs=config['student_epochs'],
        accelerator='gpu',
        devices=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            LearningRateMonitor(logging_interval='epoch')
        ],
        precision=config['precision'],
        enable_checkpointing=False,
        logger=loggers['student_stage_4'],
        fast_dev_run=config['fast_dev_run']
    )
    print('Starting stage 4')
    stage4trainer.fit(student_model, student_datamodule)

    student_checkpoint_callback = ModelCheckpoint(
        filename='student_checkpoint|epoch={epoch}|val_loss={val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_last=True,
        verbose=True
    )

    print('Starting stage 5')
    for _ in range(student_model.get_number_of_required_distillation_epochs()):
        stage5trainer = pl.Trainer(
            max_epochs=config['student_epochs'],
            accelerator='gpu',
            devices=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5),
                student_checkpoint_callback,
                LearningRateMonitor(logging_interval='epoch')
            ],
            precision=config['precision'],
            logger=loggers['student_stage_5'],
            fast_dev_run=config['fast_dev_run'],
            default_root_dir=os.path.join(config['root_output_path'], config['student_output_folder'], 'stage_5'))
        student_model.unfreeze_next_backbone_layer()
        stage5trainer.fit(student_model, student_datamodule)
    print('Distillation finished!')

    return student_model


class XtremeDistilStage1Module(pl.LightningModule):
    """Runs the representation transfer stage of Xtremedistil, i.e. tries to replicate the hidden representation of the given teacher layer with the student's hidden representation.

    Args:
        teacher: Fine-tuned teacher model.
        layer_number (int): Which layer from the teacher transformer model to distil.
        teacher_vocab (dict): Teacher model's vocabulary.
        lstm_dim (int): Hidden dimension of LSTM layer.
        lstm_layers (int): Number of LSTM layers to use.
        lstm_dropout (float): Dropout between the LSTM layers.
        student_emb_dim (int): Embedding layer vector dimension for student model.
        student_vocab (dict): Vocabulary of the student model.
        padding_idx (int): Index of the padding token for student model.
    """    
    def __init__(self, teacher, layer_number, teacher_vocab, lstm_dim, lstm_layers, lstm_dropout, student_emb_dim, student_vocab, padding_idx):
        super().__init__()

        self.teacher = teacher
        self.layer_to_distil = layer_number
        intermediate_output_dim = self.teacher.backbone._get_intermediate_layer_output_size(self.layer_to_distil)
        pretrained_embeddings = self.teacher.backbone._get_embedding_weights()

        self.student_backbone = LSTMBackbone(
            lstm_dim=lstm_dim,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            emb_dim=student_emb_dim,
            distil_vocab=student_vocab,
            pt_vocab=teacher_vocab,
            pt_embeddings=pretrained_embeddings,
            padding_idx=padding_idx,
            bidirectional=True,
            proj_dim=intermediate_output_dim
        )

    def forward(self, x):
        intermediate_teacher_out = self.teacher.get_intermediate_layer_output(x['teacher']['text_token_ids'], self.layer_to_distil)
        out = self.student_backbone(x['student']['text_token_ids']['input_ids'])

        return out, intermediate_teacher_out

    def training_step(self, batch, batch_idx):
        student_out, intermediate_teacher_out = self.forward(batch)

        loss = F.mse_loss(student_out, intermediate_teacher_out)
        self.log('loss', loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))

        # Set `T_max` to 51 after looking at BFSI training runs.
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=51, eta_min=1e-8)

        return [optimizer], [lr_scheduler]

    def get_backbone(self):
        self.freeze_backbone()
        return self.student_backbone

    def freeze_backbone(self):
        for param in self.student_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.student_backbone.parameters():
            param.requires_grad = True


class XtremeDistilStage2Module(pl.LightningModule):
    """Runs the second stage of Xtremedistil pipeline. Tries to transfer the logits of the teacher model to the student model for the training task on unlabelled data while keeping the entire student model frozen except for the final layer.

    Args:
        teacher: Fine-tuned teacher model.
        stage1_student (XtremeDistilStage1Module): Finished model from first Xtremedistil stage.
        task (str): Name of the training task.
        dropout (float): Dropout for student model.
        label_map (dict): Dictionary containing lists of all the output labels.
    """    
    def __init__(self, teacher, stage1_student, task, dropout, label_map):
        super().__init__()

        self.task = task
        self.teacher = teacher
        self.backbone = stage1_student
        backbone_output_dim = self.backbone._get_output_dim()
        self.dropout = nn.Dropout(dropout)
        self.classification_head = select_head(task)(backbone_output_dim, task, label_map)

    def forward(self, x):
        if self.task in TEXT_CLASSIFICATION_TASK_NAMES:
            out = self.dropout(self.backbone(x['student']['text_token_ids']['input_ids'])[:, -1, :])
        elif self.task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
            out = self.dropout(self.backbone(x['student']['text_token_ids']['input_ids']))
        teacher_out = self.teacher(**x['teacher']['text_token_ids'])

        return self.classification_head(out), teacher_out

    def training_step(self, batch, batch_idx):
        student_out, teacher_out = self.forward(batch)

        if self.task in TEXT_CLASSIFICATION_TASK_NAMES:
            return self.text_classification_train_step(student_out, teacher_out)
        elif self.task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
            return self.sequence_classification_train_step(student_out, teacher_out)

    def text_classification_train_step(self, student_out, teacher_out):
        loss_dict = {classname: F.mse_loss(student_out[classname], teacher_out[classname])
                        for classname in self.classification_head.get_keys()}
        aggregate_loss = torch.sum(torch.stack(list(loss_dict.values())))

        self.log('loss', aggregate_loss.item(), prog_bar=True, sync_dist=True)

        loss_logging_dict = {f'{classname}_loss': loss.item() for classname, loss in loss_dict.items()}
        self.log_dict(loss_logging_dict, prog_bar=True, sync_dist=True)

        return aggregate_loss
    
    def sequence_classification_train_step(self, student_out, teacher_out):
        loss = F.mse_loss(student_out, teacher_out)
        self.log('loss', loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))

        # Set `T_max` to 51 after looking at BFSI training runs.
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=51, eta_min=1e-8)

        return [optimizer], [lr_scheduler]


class XtremeDistilStage3Module(pl.LightningModule):
    """Runs stage 3 of the Xtremedistil pipeline. Tries to optimize student model on same task as stage 2, but while unfreezing each layer of the student one by one.

    Args:
        stage2model (XtremeDistilStage2Module): Finished model from second Xtremedistil stage.
        label_map (dict): Dictionary containing lists of all the output labels.
    """    
    def __init__(self, stage2model, label_map):
        super().__init__()
        self.model = stage2model
        self.task = stage2model.task
        self.backbone_layers = [(name, module) for name, module in self.model.backbone.named_children()]
        self.backbone_layers.reverse()
        self.backbone_layer_pointer = 0

        self.accuracy = get_multiclass_accuracy(label_map)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        student_out, teacher_out = self.forward(batch)

        if self.task in TEXT_CLASSIFICATION_TASK_NAMES:
            return self.text_classification_train_step(student_out, teacher_out)
        elif self.task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
            return self.sequence_classification_train_step(student_out, teacher_out)

    def text_classification_train_step(self, student_out, teacher_out):
        loss_dict = {classname: F.mse_loss(student_out[classname], teacher_out[classname])
                        for classname in self.model.classification_head.get_keys()}
        aggregate_loss = torch.sum(torch.stack(list(loss_dict.values())))
        self.log('loss', aggregate_loss.item(), prog_bar=True, sync_dist=True)

        loss_logging_dict = {f'{classname}_loss': loss.item() for classname, loss in loss_dict.items()}
        self.log_dict(loss_logging_dict, prog_bar=True, sync_dist=True)

        return aggregate_loss
    
    def sequence_classification_train_step(self, student_out, teacher_out):
        loss = F.mse_loss(student_out, teacher_out)
        self.log('loss', loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))

        # Set `T_max` to 51 after looking at BFSI training runs.
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=51, eta_min=1e-8)

        return [optimizer], [lr_scheduler]

    def unfreeze_next_backbone_layer(self):
        print(f'Unfreezing layer: {self.backbone_layers[self.backbone_layer_pointer][0]}')
        self.backbone_layers[self.backbone_layer_pointer][1].requires_grad_(True)
        self.backbone_layer_pointer = self.backbone_layer_pointer + 1
    
    def freeze_backbone(self):
        self.model.backbone.requires_grad_(False)

    def get_number_of_required_distillation_epochs(self):
        return len(self.backbone_layers)
    
    def get_final_model(self, label_map, samples_per_class):
        self.freeze_backbone()
        frozen_backbone = copy.deepcopy(self.model.backbone)
        if self.task in TEXT_CLASSIFICATION_TASK_NAMES:
            frozen_backbone.get_pooled_output = True
        
        classification_head = copy.deepcopy(self.model.classification_head)

        DistilledModel = select_student_model(self.task)
        final_model = DistilledModel(
            backbone=frozen_backbone,
            classification_head=classification_head,
            label_map=label_map,
            samples_per_class=samples_per_class
        )

        return final_model


def create_and_save_distilled_tokenizer(pt_tokenizer, data_path, savedir):
    """Creates and saves a Tokenizer object for the student model.

    Args:
        pt_tokenizer: Pretrained teacher model tokenizer.
        data_path (str): Path where the training data is stored.
        savedir (str): Path where to save the tokenizer vocab.

    Returns:
        Tokenizer: Tokenizer object fitted on the training data vocabulary.
    """    
    distil_tokenizer = Tokenizer(pt_tokenizer)
    data = pd.read_csv(os.path.join(data_path, 'data.csv'))
    tokenized_texts = [pt_tokenizer.tokenize(text) for text in data['text']]
    distil_tokenizer.fit(tokenized_texts)
    distil_tokenizer.save_vocabulary(
        filepath=savedir,
        filename='distil_vocab.json'
    )

    return distil_tokenizer
