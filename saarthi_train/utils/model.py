from . import TEXT_CLASSIFICATION_TASK_NAMES, SEQUENCE_CLASSIFICATION_TASK_NAMES

from ..models.text_classifiers import TransformerTextClassifier, DistilledTextClassifier
from ..models.sequence_classifiers import TransformerSequenceClassifier, DistilledSequenceClassifier
from ..models.layers import AutoClassificationOutputHead, SequenceClassificationOutputHead


def select_transformer_model(task):
    """Getter method to select the appropriate transformer model class for the given task.

    Args:
        task (str): Name of the training task.

    Returns:
        Model: Appropriate model class.
    """    
    if task in TEXT_CLASSIFICATION_TASK_NAMES:
        return TransformerTextClassifier
    elif task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
        return TransformerSequenceClassifier


def select_head(task):
    """Getter method to select the appropriate output head class for the given task.

    Args:
        task (str): Name of the training task.

    Returns:
        Head: Appropriate output head class.
    """    
    if task in TEXT_CLASSIFICATION_TASK_NAMES:
        return AutoClassificationOutputHead
    elif task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
        return SequenceClassificationOutputHead


def select_student_model(task):
    """Getter method to select the appropriate distilled model class for the given task.

    Args:
        task (str): Name of the training task.

    Returns:
        Model: Appropriate student model class.
    """    
    if task in TEXT_CLASSIFICATION_TASK_NAMES:
        return DistilledTextClassifier
    elif task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
        return DistilledSequenceClassifier
