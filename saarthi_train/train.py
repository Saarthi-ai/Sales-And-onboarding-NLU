import os
import time
import torch
from transformers import AutoTokenizer

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from .utils.model import select_transformer_model
from .utils.data import read_label_map, \
                        remove_csv_files, \
                        select_datamodule, \
                        create_output_directory, \
                        save_label_map_and_training_config
from .utils.general import get_teacher_checkpoint

from .models.distillation_modules import run_xtremedistil

from .optimizations.quantization import dynamic_quantization_for_lstms, \
                                        dynamic_quantization_for_transformers
from .optimizations.torchscript import convert_lightning_lstm_classifier_to_ts, \
                                       convert_lightning_transformer_classifier_to_ts

# Set tokenizers flag to avoid warning during runtime
os.environ['TOKENIZERS_PARALLELISM'] = "true"


def train(config, loggers):
    """Main training method.

    Args:
        config (dict): Training configuration.
        loggers (dict): Dictionary containing logger objects for each stage of training.
    """    
    label_map = read_label_map(config['input_path'])
    save_label_map_and_training_config(label_map, config, config['root_output_path'])

    create_output_directory(
        path=os.path.join(config['root_output_path'], config['teacher_output_folder']),
        already_exists_message='Teacher output directory already exists.'
    )

    teacher_checkpoint = get_teacher_checkpoint(config['teacher_name'])
    TeacherModel = select_transformer_model(config['task'])
    DataModule = select_datamodule(config['task'])

    # TODO: Test the teacher_exists branch of this conditional.
    if config['teacher_exists']:
        teacher_model = TeacherModel.from_pretrained(
            model_name=config['teacher_name'],
            task_name=config['task'],
            label_map=label_map,
            max_epochs=config['teacher_epochs'],
            total_batches=None,
            samples_per_class=[]
        )
        checkpoint = torch.load(os.path.join(config['root_output_path'], config['teacher_output_folder'], 'teacher_checkpoint.ckpt'))
        print('Found existing teacher checkpoint.')

        teacher_model.load_state_dict(checkpoint['state_dict'])
        print('Loaded teacher checkpoint.')
    else:
        teacher_datamodule = DataModule(
            path=config['input_path'],
            output_path=os.path.join(config['root_output_path'], config['teacher_output_folder']),
            task_name=config['task'],
            max_seq_len=config['max_seq_len'],
            batch_size=config['teacher_batch_size'],
            tokenizer=AutoTokenizer.from_pretrained(teacher_checkpoint)
        )
        # Dirty hack for making total_batches and samples_per_class getters work.
        teacher_datamodule.prepare_data()
        teacher_datamodule.setup(stage='fit')
        # Dirty hack over. I'm sorry you had to see this.
        total_batches = teacher_datamodule.get_total_batches()
        samples_per_class = teacher_datamodule.get_samples_per_class()
        teacher_model = TeacherModel.from_pretrained(model_name=config['teacher_name'],
                                                     task_name=config['task'],
                                                     label_map=label_map,
                                                     max_epochs=config['teacher_epochs'],
                                                     total_batches=total_batches,
                                                     samples_per_class=samples_per_class,
                                                     dropout=config['teacher_dropout'])

        teacher_callbacks = [
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=config['teacher_early_stopping_patience'],
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]

        if config['enable_teacher_checkpointing']:
            teacher_callbacks.append(ModelCheckpoint(
                dirpath=None,
                filename='teacher_checkpoint',
                save_top_k=1,
                monitor='val_loss',
                mode='min',
                verbose=True
            ))

        teacher_trainer = pl.Trainer(
            max_epochs=config['teacher_epochs'],
            accelerator='gpu',
            devices=1,
            callbacks=teacher_callbacks,
            gradient_clip_val=config['teacher_clip_grad_norm'],
            gradient_clip_algorithm='norm',
            enable_checkpointing=config['enable_teacher_checkpointing'],
            precision=config['precision'],
            logger=loggers['teacher'],
            fast_dev_run=config['fast_dev_run'],
            default_root_dir=os.path.join(config['root_output_path'], config['teacher_output_folder'])
        )

        teacher_trainer.fit(teacher_model, teacher_datamodule)

        teacher_output_path = os.path.join(config['root_output_path'], config['teacher_output_folder'], 'lightning_logs', 'version_0')

        if config['enable_teacher_checkpointing']:
            if config['quantization']:
                teacher_model = dynamic_quantization_for_transformers(teacher_model, config['quantization'])
                config['torchscript'] = True  # If model has been quantized, torchscript is required for saving it.

            if config['torchscript']:
                teacher_model.eval()
                convert_lightning_transformer_classifier_to_ts(teacher_model, config, os.path.join(teacher_output_path, 'checkpoints', 'model.pt'))

        # Removing artifacts not required for deployment.
        if config['production'] and config['enable_teacher_checkpointing']:
            for item in os.listdir(teacher_output_path):
                if not os.path.isdir(os.path.join(teacher_output_path, item)):
                    os.remove(os.path.join(teacher_output_path, item))

            if config['torchscript']:
                os.remove(os.path.join(teacher_output_path, 'checkpoints', 'teacher_checkpoint.ckpt'))

        # Preprocessed data cleanup
        remove_csv_files(os.path.join(config['root_output_path'], config['teacher_output_folder']))

        print('Teacher finetuning finished!')

    if config['distillation']:
        create_output_directory(
            path=os.path.join(config['root_output_path'], config['student_output_folder']),
            already_exists_message='Student directory already exists.'
        )

        distilled_model = run_xtremedistil(config, teacher_model, label_map, loggers)

        student_output_path = os.path.join(config['root_output_path'], config['student_output_folder'], 'lightning_logs', 'version_0')

        if config['quantization']:
            distilled_model = dynamic_quantization_for_lstms(distilled_model, dtype=config['quantization'])
            config['torchscript'] = True  # If model has been quantized, torchscript is required for saving it.

        if config['torchscript']:
            convert_lightning_lstm_classifier_to_ts(distilled_model, config, os.path.join(student_output_path, 'checkpoints', 'model.pt'))

        if config['production'] and not config['fast_dev_run']:
            for item in os.listdir(student_output_path):
                if not os.path.isdir(os.path.join(student_output_path, item)):
                    os.remove(os.path.join(student_output_path, item))

            for item in os.listdir(os.path.join(student_output_path, 'checkpoints')):
                if config['torchscript']:
                    if item.endswith('.ckpt'):
                        os.remove(os.path.join(student_output_path, 'checkpoints', item))
                else:
                    if item != 'last.ckpt':
                        os.remove(os.path.join(student_output_path, 'checkpoints', item))

        remove_csv_files(os.path.join(config['root_output_path'], config['student_output_folder']))


if __name__ == '__main__':
    start = time.time()
    args = {
        'input_path': './nlu_inputs/bfsi_sales_and_onboarding/',
        'root_output_path': './nlu_outputs/bfsi_sales_and_onboarding/',
        'max_seq_len': 26,
        'precision': '16-mixed',
        'task': 'intent_classification', #intent_classification, sequence_classification for NLU ner for NER
        'teacher_name': 'xlm-roberta',
        'teacher_epochs': 3,
        'teacher_batch_size': 16,
        'teacher_dropout': 0.1,
        'teacher_clip_grad_norm': None,
        'teacher_early_stopping_patience': 2,
        'enable_teacher_checkpointing': True,
        'teacher_output_folder': 'teacher',
        'torchscript': False,
        'quantization': None,
        'teacher_exists': False,
        'distillation': True,
        'teacher_layer_to_distil': 6,
        'student_lstm_dim': 600,
        'student_lstm_layers': 1,
        'student_lstm_dropout': 0.3,
        'student_word_emb_dim': 300,
        'student_epochs': 200,
        'student_batch_size': 512,
        'student_output_folder': 'student',
        'fast_dev_run': False,
        'production': False
    }
    teacher_tensorboard_logger = TensorBoardLogger(save_dir=os.path.join(args['root_output_path'], args['teacher_output_folder']))
    student_tensorboard_logger = TensorBoardLogger(save_dir=os.path.join(args['root_output_path'], args['student_output_folder']))
    loggers = {
        'teacher': [teacher_tensorboard_logger],
        'student_stage_1': [student_tensorboard_logger],
        'student_stage_2': [student_tensorboard_logger],
        'student_stage_3': [student_tensorboard_logger],
        'student_stage_4': [student_tensorboard_logger],
        'student_stage_5': [student_tensorboard_logger]
    }

    train(args, loggers)
    end = time.time() - start

    print(f'Time taken: {end / 60} mins')
