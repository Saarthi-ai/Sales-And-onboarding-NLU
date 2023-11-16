import os
import json
import torch

from .data.preprocessing import remove_punctuations_and_convert_to_lowercase, remove_punctuations_and_convert_to_lowercase_ner
from .inference.model_builder import get_finetuned_teacher, get_finetuned_student
from .postprocessing.main import postprocess_model_output
from .utils import TEXT_CLASSIFICATION_TASK_NAMES
from .postprocessing.utils import TextToInt


def init(output_path, model_type='student'):
    """Initializes model and required dependencies for inference.

    Args:
        output_path (str): Path where the trained model is stored.
        model_type (str, optional): Type of model to be initialized. Can be either 'teacher' or 'student'. Defaults to 'student'.
    """    
    global label_map
    global model
    global tokenizer
    global max_seq_len
    global task
    global task_type

    with open(os.path.join(output_path, 'labels.json'), 'r') as f:
        label_map = json.load(f)

    
    model, tokenizer, max_seq_len, task = eval(f'get_finetuned_{model_type}(output_path)')
    task_type = 'text_classification' if task in TEXT_CLASSIFICATION_TASK_NAMES else 'sequence_classification'


def run(text, lang=None, **kwargs):
    """Runs the inference pipeline on the given input text.

    Args:
        text (str): Input to the model.
        lang (str, optional): Language of the input text. Defaults to None.

    Returns:
        dict: Final postprocessed output from the model.
    """    
    return eval(f'run_{task_type}(text, lang, **kwargs)')


# lang and **kwargs kept here for consistency. They don't get used.
def run_text_classification(text, lang, **kwargs):
    """Runs the text classification inference pipeline on the given input text.

    Args:
        text (str): Input text to the model.

    Returns:
        dict: Final postprocessed output from the model.
    """    
    model_in = tokenizer(
        [remove_punctuations_and_convert_to_lowercase(text)],
        padding='max_length',
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt'
    )

    with torch.no_grad():
        out = model(model_in['input_ids'])

    #return postprocess_model_output(
    #    text=None,
    #    lang=None,
    #    raw_model_out=out,
    #    label_map=label_map,
    #    task=task,
    #    offset_mapping=None
    #)

    return {
        'response' : postprocess_model_output(
            text=None,
            lang=None,
            raw_model_out=out,
            label_map=label_map,
            task=task,
            offset_mapping=None
        )
    }



def run_sequence_classification(text, lang, **kwargs):
    """Runs the sequence classification inference pipeline on the given input.

    Args:
        text (str): Input to the model.
        lang (str): Language of the input text. Required for postprocessing.

    Returns:
        dict: Final postprocessed output from the model.
    """    
    text = remove_punctuations_and_convert_to_lowercase_ner(text)
    
    # Adding ITN in preprocessing
    config_folder = "./saarthi_train/ner_postprocessing_config"
    sub_config_path = os.path.join(config_folder, "subs.json")
    itn_std_path = os.path.join(config_folder, "itn_standardizations_v4.json")
    textToInt = TextToInt(sub_config_path, 
                          itn_std_path) 
    normalized_text = textToInt.convert_text_to_int(" " + text + " ", use_itn=True, lang = lang)
    model_in = tokenizer(
        [normalized_text],
        padding='max_length',
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
        return_tensors='pt'
    )


    with torch.no_grad():
        out = model(model_in['input_ids'])

    

    return postprocess_model_output(
        text=text,
        lang=lang,
        raw_model_out=out,
        label_map=label_map,
        task=task,
        offset_mapping=model_in['offset_mapping'][0],
        normalized_text=normalized_text,
        **kwargs
    )
