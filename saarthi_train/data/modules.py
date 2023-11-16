import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .preprocessing import preprocess_df, preprocess_sequence_df
from .datasets import TextClassificationDataset, \
                      SequenceClassificationDataset, \
                      RepresentationTransferDataset


# TODO: Refactor TextClassification and SequenceClassification by creating a BaseText class
# Which takes a Dataset class, and preprocessing function as input.
# Inherit the base class for the Text and Sequence Classification modules.
class TextClassificationDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule class for text classification workloads.

    Args:
        path (str): Path where the data files are present.
        output_path (str): Output path.
        max_seq_len (int): Maximum sequence length of the model.
        tokenizer: Model's tokenizer.
        task_name (str): Name of the task. For ex - intent_classification, text_classification.
        batch_size (int): Batch size to be used during model training. Defaults to 64.
    """    
    def __init__(self, path, output_path, max_seq_len, tokenizer, task_name, batch_size=64):
        # Including task_name parameter for consistency across datamodule APIs.
        super().__init__()
        self.data_dir = path
        self.output_path = output_path

        with open(os.path.join(self.data_dir, 'labels.json'), 'r') as f:
            self.label_map = json.load(f)

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def prepare_data(self):
        data = pd.read_csv(os.path.join(self.data_dir, 'data.csv'))
        result = preprocess_df(data, self.label_map, self.max_seq_len, self.tokenizer)
        result.to_csv(os.path.join(self.output_path, 'train_preprocessed.csv'), index=False)

        if os.path.isfile(os.path.join(self.data_dir, 'test.csv')):
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            test_preprocessed = preprocess_df(test_df, self.label_map, self.max_seq_len, self.tokenizer)
            test_preprocessed.to_csv(os.path.join(self.output_path, 'test_preprocessed.csv'))

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            data = pd.read_csv(os.path.join(self.output_path, 'train_preprocessed.csv'))
            train, val = train_test_split(data, test_size=0.1)
            self.train_dataset = TextClassificationDataset(train)
            self.val_dataset = TextClassificationDataset(val)

        if stage in (None, 'test'):
            test = pd.read_csv(os.path.join(self.output_path, 'test_preprocessed.csv'))
            self.test_dataset = TextClassificationDataset(test)

    def train_dataloader(self):
        return self.create_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(dataset=self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.create_dataloader(dataset=self.test_dataset, shuffle=False)

    def create_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count()
        )

    def get_total_batches(self):
        """Returns the total number of training batches.

        Returns:
            int: Total number of batches that will be used during training.
        """
        return len(self.train_dataset) // self.batch_size

    def get_samples_per_class(self):
        """Returns the number of samples per class for each output of the model.

        Returns:
            list: Samples per class per output.
        """
        return self.train_dataset._samples_per_class()


class SequenceClassificationDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule class for sequence classification workloads.

    Args:
        path (str): Path where the data files are present.
        output_path (str): Output path.
        max_seq_len (int): Maximum sequence length of the model.
        tokenizer: Model's tokenizer.
        task_name (str): Name of the task. For ex - ner, pos, etc.
        batch_size (int): Batch size to be used during model training. Defaults to 64.
    """    
    def __init__(self, path, output_path, max_seq_len, tokenizer, task_name, batch_size=64):
        super().__init__()
        self.data_dir = path
        self.output_path = output_path
        self.task_name = task_name

        with open(os.path.join(self.data_dir, 'labels.json'), 'r') as f:
            self.label_map = json.load(f)

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def prepare_data(self):
        data = pd.read_csv(os.path.join(self.data_dir, 'data.csv'))
        data.drop(columns = ['intent'], inplace = True)
        result = preprocess_sequence_df(data, self.task_name, self.label_map, self.max_seq_len, self.tokenizer)
        result.to_csv(os.path.join(self.output_path, f'train_{self.task_name}_preprocessed.csv'), index=False)

        if os.path.isfile(os.path.join(self.data_dir, 'test.csv')):
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            test_df.drop(columns = ['intent'], inplace = True)
            test_preprocessed = preprocess_sequence_df(test_df, self.task_name, self.label_map, self.max_seq_len, self.tokenizer)
            test_preprocessed.to_csv(os.path.join(self.output_path, f'test_{self.task_name}_preprocessed.csv'))

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            data = pd.read_csv(os.path.join(self.output_path, f'train_{self.task_name}_preprocessed.csv'))
            train, val = train_test_split(data, test_size=0.1)
            self.train_dataset = SequenceClassificationDataset(train, self.task_name)
            self.val_dataset = SequenceClassificationDataset(val, self.task_name)

        if stage in (None, 'test'):
            test = pd.read_csv(os.path.join(self.output_path, f'test_{self.task_name}_preprocessed.csv'))
            self.test_dataset = SequenceClassificationDataset(test, self.task_name)

    def train_dataloader(self):
        return self.create_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(dataset=self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.create_dataloader(dataset=self.test_dataset, shuffle=False)

    def create_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count()
        )

    def get_total_batches(self):
        """Returns the total number of training batches.

        Returns:
            int: Total number of batches that will be used during training.
        """
        return len(self.train_dataset) // self.batch_size

    def get_samples_per_class(self):
        """Returns the number of samples per class for each output of the model.

        Returns:
            int: Samples per class per output.
        """
        return self.train_dataset._samples_per_class()


class RepresentationTransferDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule class for first stage of XtremeDistil.

    Args:
        path (str): Path where the data files are present.
        output_path (str): Output path.
        label_map (dict): Dictionary containing list of all the labels for the training task.
        max_seq_len (int): Maximum sequence length of the model.
        pt_tokenizer: Teacher model's pretrained tokenizer object.
        distil_tokenizer (Tokenizer): Student model's tokenizer object.
        batch_size (int): Batch size to be used during model training. Defaults to 64.
    """    
    def __init__(self, path, output_path, label_map, max_seq_len, pt_tokenizer, distil_tokenizer, batch_size=64):
        super().__init__()
        self.data_dir = path
        self.output_path = output_path
        self.label_map = label_map

        self.max_seq_len = max_seq_len
        self.pt_tokenizer = pt_tokenizer
        self.distil_tokenizer = distil_tokenizer
        self.batch_size = batch_size

    def prepare_data(self):
        data = pd.read_csv(os.path.join(self.data_dir, 'transfer.csv')).astype(str)

        result_teacher = preprocess_df(data, self.label_map, self.max_seq_len, self.pt_tokenizer)
        result_teacher.to_csv(os.path.join(self.output_path, 'transfer_preprocessed_teacher.csv'), index=False)

        result_student = preprocess_df(data, self.label_map, self.max_seq_len, self.distil_tokenizer)
        result_student.to_csv(os.path.join(self.output_path, 'transfer_preprocessed_student.csv'), index=False)

    def setup(self, stage):
        teacher_data = pd.read_csv(os.path.join(self.output_path, 'transfer_preprocessed_teacher.csv'))
        student_data = pd.read_csv(os.path.join(self.output_path, 'transfer_preprocessed_student.csv'))
        self.train_dataset = RepresentationTransferDataset(teacher_data, student_data)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count()
        )
