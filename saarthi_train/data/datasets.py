import json
import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """PyTorch Dataset class for text classification workloads.

    Args:
        df (pandas.DataFrame): Preprocessed dataframe containing the training data for a text classification model.
    """    
    def __init__(self, df):
        self.data = df
        for col in self.data:
            if 'token_ids' in col:
                self.data[col] = self.data[col].str.replace('\'', '"')
                self.data[col] = self.data[col].map(json.loads)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {classname: value for classname, value in zip(self.data.keys(), self.data.iloc[idx])}

    def collate_fn(self, data):
        batch = {}

        for key in data[0].keys():
            if 'token_ids' in key:
                token_contents = {}
                token_keys = data[0][key].keys()
                for token_key in token_keys:
                    token_contents[token_key] = torch.tensor([item[key][token_key] for item in data])
                batch[key] = token_contents
            else:
                batch[key] = torch.tensor([item[key] for item in data])

        return batch
    
    def _samples_per_class(self):
        counts = {}
        for classname in self.data:
            if 'token_ids' in classname:
                continue
            class_counts = dict(self.data[classname].value_counts())
            result = [0] * len(class_counts)
            for value, count in class_counts.items():
                result[value] = count
            counts[classname] = result

        return counts

class SequenceClassificationDataset(Dataset):
    """PyTorch Dataset class for sequence classification workloads.

    Args:
        df (pandas.DataFrame): Preprocessed dataframe containing the training data for a text classification model.
        task_name (str): Name of the sequence classification task. For ex. - ner, pos, etc.
    """    
    def __init__(self, df, task_name):
        self.data = df
        self.task_name = task_name
        for col in self.data:
            if 'token_ids' in col:
                self.data[col] = self.data[col].str.replace('\'', '"')
            self.data[col] = self.data[col].map(json.loads)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {classname: value for classname, value in zip(self.data.keys(), self.data.iloc[idx])}

    def collate_fn(self, data):
        batch = {}
        for key in data[0].keys():
            if 'token_ids' in key:
                token_contents = {}
                token_keys = data[0][key].keys()
                for token_key in token_keys:
                    token_contents[token_key] = torch.tensor([item[key][token_key] for item in data])
                batch[key] = token_contents
            else:
                batch[key] = torch.tensor([item[key] for item in data])

        return batch

    def _samples_per_class(self):
        counts = dict(self.data[self.task_name].explode().value_counts())
        result = [0] * len(counts)
        for value, count in counts.items():
            result[value] = count

        return result


class RepresentationTransferDataset(Dataset):
    """PyTorch Dataset class for the representation transfer stage of XtremeDistil knowledge distillation process.

    Args:
        teacher_df (pandas.DataFrame): Pandas dataframe containing preprocessed unlabelled data for the teacher model.
        student_df (pandas.DataFrame): Pandas dataframe containing preprocessed unlabelled data for the student model.
    """    
    def __init__(self, teacher_df, student_df):
        self.teacher_data = self._convert_dtype_from_string(teacher_df)
        self.student_data = self._convert_dtype_from_string(student_df)

    def __len__(self):
        return len(self.teacher_data)

    def __getitem__(self, idx):
        return {model: {key: value for key, value in zip(data.keys(), data.iloc[idx])} for model, data in [('teacher', self.teacher_data), ('student', self.student_data)]}

    def collate_fn(self, data):
        batch = {}

        for target_model, model_inputs in data[0].items():
            model_batch = {}
            for key in model_inputs:
                contents = {}
                token_keys = model_inputs[key].keys()
                for token_key in token_keys:
                    contents[token_key] = torch.tensor([item[target_model][key][token_key] for item in data])
                model_batch[key] = contents
            batch[target_model] = model_batch

        return batch

    def _convert_dtype_from_string(self, df):
        for col in df:
            df[col] = df[col].str.replace('\'', '"')
            df[col] = df[col].map(json.loads)

        return df
