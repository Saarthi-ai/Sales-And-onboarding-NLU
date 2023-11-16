import pytz
from cmath import nan
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from .entities_helper import Entity
from datetime import datetime as Date
from .event_list import EventExtraction
from .OffsetHelper import GrainUnitExtraction
from .rules_utils.rules import RulesRefactored
from .time_utils import DateHelperRefactor, TimeHelper
from .utils import get_ner_prediction_text_new, get_minimum_valid_value, get_reverse_time_module
from .entities_helper import IdentifiedEntities, PostprocessingConfig, assign_values_to_entities_class, \
                                get_date, get_time, get_relation, get_entity
from ..utils import TEXT_CLASSIFICATION_TASK_NAMES, SEQUENCE_CLASSIFICATION_TASK_NAMES


def postprocess_model_output(text, lang, raw_model_out, label_map, task, offset_mapping=None, normalized_text = None, **kwargs):
    """Postprocesses model output.

    Args:
        text (str): Input text to model.
        lang (str): Language of the input text.
        raw_model_out (torch.Tensor): Raw model output for input text.
        label_map (dict): Dictionary containing lists of all the output labels.
        task (str): Task that the model was trained on.
        offset_mapping (dict, optional): Offset mapping for sub-word tokens. Defaults to None.
        normalized_text (str, optional): Inverse text normalized output. Defaults to None.

    Returns:
        dict: Final postprocessed model output.
    """    
    if task in TEXT_CLASSIFICATION_TASK_NAMES:
        probs = {classname: torch.topk(F.softmax(output_tensor, dim=-1), k=2, dim=-1) 
                 for classname, output_tensor in raw_model_out.items()}
        return {classname: [{'name': label_map[classname][pred.item()], 'confidence': float(confidence.item())} 
                              for confidence, pred in zip(topk[0][0], topk[1][0])] 
                              for classname, topk in probs.items()}
    elif task in SEQUENCE_CLASSIFICATION_TASK_NAMES:
        probs = F.softmax(raw_model_out, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        final_tags = aggregate_subword_tags(offset_mapping, [label_map[task][pred] for pred in preds[0]])        
        entities = postprocess_entities(" " + normalized_text + " ", final_tags, lang, **kwargs)

        entity_output = []
        for entity_type, entity_values in entities.items():
            entity_dict = {}
            entity_dict['value'] = []
            for key, value in entity_values.items():
                entity_dict[key] = value
            if 'value' in entity_dict and len(entity_dict['value'])>0:
                entity_dict['value'] = entity_dict['value'][0]
                entity_dict['entity'] = entity_type
                entity_dict['extractor'] = 'saarthi-ner'
                entity_output.append(
                    entity_dict
                )

        return {
            "text": text,
            "normalized_text": normalized_text,
            "entities": entity_output,
            "tags": final_tags
        }


def get_festival(ptp_date: str, current_date: Date, lang: str) -> Tuple[Date, bool]:
    """
    Function to identity festival date in the text 

    Args: 
        ptp_date (str): Text which ner model tags
        lang (str): Language parameter
    
    Returns:
        Tuple[Date, bool]: Festival date or current date based on whether festival date is available in text or not.
    """
    festival_bool = False
    
    for text in ptp_date:
        text = " " + text + " "
        eventExtraction = EventExtraction()
        date_festival = eventExtraction.extract_festivals(text, current_date, lang)
        if date_festival != -1:
            current_date = Date.strptime(date_festival, '%d/%m/%Y')
            current_date = current_date.replace(tzinfo = pytz.timezone('Asia/Kolkata'))
            festival_bool = True
    return current_date, festival_bool


def reference_date_finder(ptp_date: str, current_date: str=None, lang: str='hindi') -> Tuple[Date, bool]:
    """
    Function to change the reference dates

    Args: 
        ptp_date (str): Text which ner model tags
        current_date (str): Current date parameter or reference date if it comes from payload
        lang (str): Language parameter
    
    Returns:
        Tuple[Date, bool]: Any reference date based on whether date is given in payload or not.
    """
    if current_date == None or current_date == nan:
        current_date = Date.now(pytz.timezone('Asia/Kolkata'))
    else:
        current_date = Date.strptime(current_date, '%d/%m/%Y %H:%M:%S')
        current_date = current_date.replace(tzinfo = pytz.timezone('Asia/Kolkata'))
    
    current_date, festival_bool = get_festival(ptp_date, current_date, lang)

    return current_date, festival_bool

def define_ambigous_date(due_date: str, weekend: str, new_temp: List[Date], consider_due_date: bool) -> Union[Date, str]:
    ambiguos_dates = ""
    if due_date == "due_date":
        ambiguos_dates = "due_date"
    elif weekend == "weekend":
        ambiguos_dates = "weekend"
    elif weekend == "next_weekend":
        ambiguos_dates = "next_weekend"
    elif abs(min(new_temp))>0:
        if consider_due_date:
            ambiguos_dates = "due_date"
        else:
            ambiguos_dates = Date.now(pytz.timezone('Asia/Kolkata'))    
    return ambiguos_dates

def postprocess_entities(utterance: str,
                         labels: List[str],
                         lang: str = 'hindi',
                         **kwargs):
    kwargs = kwargs['request']
    consider_word_cnt = kwargs["consider_word_cnt"] if "consider_word_cnt" in kwargs else 0
    consider_due_date = kwargs["consider_due_date"] if "consider_due_date" in kwargs else True
    consider_weekend = kwargs["consider_weekend"] if "consider_weekend" in kwargs else False
    back_date = kwargs["back_date"] if "back_date" in kwargs else False
    current_date = kwargs["date"] if "date" in kwargs else None

    entities_list = [entity.value for entity in Entity]

    ner_out = get_ner_prediction_text_new(utterance, labels, entities_list)
    current_date, festival_bool = reference_date_finder(ner_out['ptp_date'], current_date, lang)

    postprocessingConfig = PostprocessingConfig(
                                                utterance=utterance, \
                                                labels=labels, \
                                                current_date=current_date, \
                                                lang = lang, \
                                                consider_word_cnt = consider_word_cnt, \
                                                consider_due_date = consider_due_date, \
                                                consider_weekend = consider_weekend, \
                                                back_date = back_date, \
                                                festival_bool = festival_bool
                                            )

    grain_variants_file = './saarthi_train/ner_postprocessing_config/new_grains.json'
    grainUnitExtraction = GrainUnitExtraction(grain_variants_file=grain_variants_file)
    rulesRefactored = RulesRefactored(lang, grainUnitExtraction = grainUnitExtraction)
    entities = IdentifiedEntities()

    temp, number, due_date, weekend = get_date(grainUnitExtraction, postprocessingConfig, \
                                                rulesRefactored, ner_out["ptp_date"])
    entities.number['value'] = number
    valid_dates = [pred for pred in temp if pred != "none"]

    if "relation" in ner_out:
        relation = get_relation(rulesRefactored, ner_out["relation"], postprocessingConfig.lang, "relation")
        if len(relation) == 0:
            if "person" in ner_out:
                ner_out["person"].extend(ner_out["relation"])
        entities.relation['value'] = relation
    
    if "person" in ner_out:
        entities.person['value'] = ner_out['person']
    
    if "location" in ner_out:
        entities.location['value'] = get_entity(rulesRefactored, ner_out["location"], utterance, \
                                        postprocessingConfig.lang, postprocessingConfig.consider_word_cnt, "location")

    if "organisation" in ner_out:    
        entities.organisation['value'] = get_entity(rulesRefactored, ner_out["organisation"], utterance, \
                                        postprocessingConfig.lang, postprocessingConfig.consider_word_cnt, "organisation")

    if "vehicle_brand" in ner_out:
        entities.vehicle_brand['value'] = get_entity(rulesRefactored, ner_out["vehicle_brand"], utterance, \
                                        postprocessingConfig.lang, postprocessingConfig.consider_word_cnt, "vehicle_brand")

    if "occupation" in ner_out:
        entities.occupation["value"] = get_entity(rulesRefactored, ner_out["occupation"], utterance, \
                                        postprocessingConfig.lang, postprocessingConfig.consider_word_cnt, "occupation") 

    if "qualification" in ner_out:
        entities.qualification["value"] = get_entity(rulesRefactored, ner_out["qualification"], utterance, \
                                        postprocessingConfig.lang, postprocessingConfig.consider_word_cnt, "qualification")    

    if len(valid_dates) != 0:
        date = min(valid_dates)
        only_date = min(valid_dates)
    else:
        date = current_date
        only_date = "none"

    time_str = ner_out["time"]
    if len(time_str) == 0 and len(valid_dates) != 0:
        # if not has_string_element(valid_dates):
        try:
            entities.date['value'] = [DateHelperRefactor.convert_datetime_to_string(min(valid_dates))]
            entities.time['value'] = []
            return entities.__dict__
        except Exception as ex:
            ambiguos_dates = define_ambigous_date(due_date=due_date, weekend=weekend, new_temp=valid_dates, consider_due_date=consider_due_date)
            entities.date['value'] = assign_values_to_entities_class(
                    consider_due_date = consider_due_date,
                    ambiguos_dates=ambiguos_dates,
                    dates= valid_dates,
                    only_date = only_date
                )
            entities.time['value'] = []
            return entities.__dict__
                
        
    temp = get_time(grainUnitExtraction, rulesRefactored, date, time_str, lang)


    valid_time, valid_time_module = [], []
    for pred, pred_time_module in temp:
        if pred != "none":
            valid_time.append(pred)
            valid_time_module.append(pred_time_module)

    if len(valid_time) != 0:
        time, time_module = get_minimum_valid_value((valid_time, valid_time_module))
        # time = min(valid_time)
        time = TimeHelper.validate_time(time, utterance, grainUnitExtraction = grainUnitExtraction, lang = lang)
        # if isinstance(time, Date):
        try:
            entities.date['value'] = [DateHelperRefactor.convert_datetime_to_string(time)]
            entities.time['value'] = [TimeHelper.convert_datetime_to_string(time)]
            #TODO change the code here to take the time module from config files
            entities.time['preposition'] = get_reverse_time_module(time_module)
            return entities.__dict__
        except Exception as ex:
            ambiguos_dates = define_ambigous_date(due_date=due_date, weekend=weekend, new_temp=valid_time, consider_due_date=consider_due_date)
            entities.date['value'] = assign_values_to_entities_class(
                    consider_due_date = consider_due_date,
                    ambiguos_dates=ambiguos_dates,
                    dates = valid_time,
                    only_date=only_date
                )
            entities.time['value'] = []
            return entities.__dict__
    else:
        try:
            # if not has_string_element(only_date):
            try:
                entities.date['value'] = [DateHelperRefactor.convert_datetime_to_string(only_date)]
                entities.time['value'] = []
                return entities.__dict__           
            except Exception as ex:
                ambiguos_dates = define_ambigous_date(due_date=due_date, weekend=weekend, new_temp=valid_time, consider_due_date=consider_due_date)
                entities.date['value'] = assign_values_to_entities_class(
                    consider_due_date = consider_due_date,
                    ambiguos_dates=ambiguos_dates,
                    dates = valid_time,
                    only_date=only_date
                )
                entities.time['value'] = []
                return entities.__dict__
        except:
            entities.date['value'] = []
            entities.time['value'] = []
            return entities.__dict__

def aggregate_subword_tags(offset_mapping, subword_tags):
        # Initialize result and current_word_tag
        result = []
        current_word_tag = None
        prev_end = -1  # Track the end of the previous offset

        for (start, end), tag in zip(offset_mapping, subword_tags):
            # Skip if it's a special token
            if start == 0 and end == 0:
                continue

            # If the tag starts with 'U-'
            if tag.startswith('U-'):
                if current_word_tag:
                    result.append(current_word_tag)
                    current_word_tag = None
                result.append(tag)
                continue

            # If it's a beginning of a new word
            if start != prev_end:
                if current_word_tag:
                    result.append(current_word_tag)
                current_word_tag = tag
            # If it's a subword
            elif tag.startswith('B-') or (tag.startswith('I-') and not current_word_tag.startswith('B-')):
                current_word_tag = tag

            prev_end = end

        # Append the last word tag if it exists
        if current_word_tag:
            result.append(current_word_tag)

        return result


if __name__=='__main__':
    text = "10 दिन के बाद में"
    entities = ["B-date", 'I-date', 'I-date', 'L-date']
    print(postprocess_entities(text, entities))
