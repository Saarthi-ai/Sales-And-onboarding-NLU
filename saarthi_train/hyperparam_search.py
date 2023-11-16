import os
import json
import optuna

from lightning.pytorch.loggers import TensorBoardLogger

from .train import train
from .utils.loggers import OptunaLogger


def objective(trial):
    args = {
        'input_path': './inputs/ner',
        'root_output_path': './outputs/ner',
        'max_seq_len': 26,
        'precision': '16-mixed',
        'task': 'ner',
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
        'distillation': False,
        'teacher_layer_to_distil': 6,
        'student_lstm_dim': trial.suggest_int('student_lstm_dim', 50, 600),
        'student_lstm_layers': 1,
        'student_lstm_dropout': trial.suggest_float('student_lstm_dropout', 0.1, 0.5),
        'student_word_emb_dim': trial.suggest_int('student_word_emb_dim', 50, 300),
        'student_epochs': 200,
        'student_batch_size': 512,
        'student_output_folder': 'student',
        'fast_dev_run': False,
        'production': False
    }
    optuna_logger = OptunaLogger()
    tensorboard_logger = TensorBoardLogger(save_dir=os.path.join(args['root_output_path'], args['student_output_folder']), name='default_logger')
    loggers = {
        'teacher': [tensorboard_logger],
        'student_stage_1': [tensorboard_logger],
        'student_stage_2': [tensorboard_logger],
        'student_stage_3': [tensorboard_logger],
        'student_stage_4': [tensorboard_logger],
        'student_stage_5': [optuna_logger],
    }
    train(args, loggers)

    return optuna_logger.cache['val_loss'][-1]


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    with open('best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print('Hyperparameter search finished!')
    print(f'Best parameters: {study.best_params}')
