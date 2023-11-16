import os
import sys
import argparse

from train import train
from utils.loggers import AzureMLLogger

from azureml.core import Run
from azureml.core import Workspace, Dataset


os.environ['TOKENIZERS_PARALLELISM'] = "true"
run = Run.get_context()

parser = argparse.ArgumentParser()

# Global arguments
parser.add_argument('--max_seq_len', type=int, default=26, required=True)
parser.add_argument('--root_output_path', type=str, required=True)
parser.add_argument('--precision', type=str, default='32-true', required=False)

# Dataset arguments
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--dataset_version', type=int, required=True)

# Teacher arguments
parser.add_argument('--teacher_name', type=str, required=True)
parser.add_argument('--teacher_epochs', type=int, default=80, required=True)
parser.add_argument('--teacher_batch_size', type=int, default=16, required=True)
parser.add_argument('--teacher_dropout', type=float, default=0.1, required=True)
parser.add_argument('--teacher_clip_grad_norm', default=None, required=False)
parser.add_argument('--enable_teacher_checkpointing', type=bool, default=False, required=False)
parser.add_argument('--teacher_output_folder', type=str, required=True)
parser.add_argument('--teacher_exists', type=bool, default=False, required=False)

# Student arguments
parser.add_argument('--distillation', type=bool, default=False, required=False)
parser.add_argument('--teacher_layer_to_distil', type=int, default=6, required=False)
parser.add_argument('--student_lstm_dim', type=int, default=600, required=False)
parser.add_argument('--student_lstm_layers', type=int, default=1, required=False)
parser.add_argument('--student_lstm_dropout', type=float, default=0, required=False)
parser.add_argument('--student_word_emb_dim', type=int, default=300, required=False)
parser.add_argument('--student_epochs', type=int, default=200, required=False)
parser.add_argument('--student_batch_size', type=int, default=64, required=False)
parser.add_argument('--student_output_folder', type=str, required=False)

# Miscellaneous arguments
parser.add_argument('--fast_dev_run', type=bool, default=False, required=False)
parser.add_argument('--production', type=bool, default=True, required=True)


def download_dataset(name, version):
    workspace = run.experiment.workspace
    dataset = Dataset.get_by_name(workspace, name=name, version=version)
    dataset.download(target_path=name, overwrite=False)


def main(argv):
    args = vars(parser.parse_args(argv[1:]))
    args['input_path'] = args['dataset_name']

    print('Training configuration:')
    print(args)
    
    download_dataset(args['dataset_name'], args['dataset_version'])

    azureml_logger = AzureMLLogger(run)
    loggers = {
        'teacher': [azureml_logger],
        'student_stage_1': [azureml_logger],
        'student_stage_2': [azureml_logger],
        'student_stage_3': [azureml_logger],
        'student_stage_4': [azureml_logger],
        'student_stage_5': [azureml_logger],
    }
    train(args, loggers=loggers)


if __name__ == '__main__':
    main(sys.argv)
