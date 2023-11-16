import os
import json

from . import TEXT_CLASSIFICATION_TASK_NAMES, SEQUENCE_CLASSIFICATION_TASK_NAMES
from ..data.modules import TextClassificationDataModule, SequenceClassificationDataModule


def read_label_map(path):
    """Reads label map json file.

    Args:
        path (str): Path to label map json file.

    Returns:
        dict: Label map
    """
    with open(os.path.join(path, 'labels.json'), 'r') as f:
        label_map = json.load(f)

    return label_map


def save_label_map_and_training_config(label_map, training_config, output_path):
    """Saves label map and training config as json.

    Args:
        label_map (dict): Dictionary containing lists of all the output labels.
        training_config (dict): Training configuration.
        output_path (str): Path to save the files to.
    """
    with open(os.path.join(output_path, 'labels.json'), 'w') as f:
        json.dump(label_map, f)

    with open(os.path.join(output_path, 'training_config.json'), 'w') as f:
        json.dump(training_config, f)


def create_output_directory(path, already_exists_message):
    """Create output directory in the filesystem.

    Args:
        path (str): Path of the folder to create.
        already_exists_message (str): Error message to be displayed if the directory already exists.
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        print(already_exists_message)


def select_datamodule(task):
    """Returns datamodule class based on input task.

    Args:
        task (str): Name of training task.

    Returns:
        DataModule: Relevant DataModule class.
    """
    if task in TEXT_CLASSIFICATION_TASK_NAMES:
        return TextClassificationDataModule
    elif task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
        return SequenceClassificationDataModule


def remove_csv_files(path):
    """Delete csv files from the given path.

    Args:
        path (str): Path from where the .csv files need to be cleaned.
    """
    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
    for file in csv_files:
        os.remove(os.path.join(path, file))
        print(f'Cleaned up file: {file}')
