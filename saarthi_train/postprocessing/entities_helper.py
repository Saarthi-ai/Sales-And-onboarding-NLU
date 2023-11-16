from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime as Date, timedelta
from .time_utils import DateHelperRefactor, TimeHelper
import re
from .rules_utils.rules import RulesRefactored
from .OffsetHelper import GrainUnitExtraction
from enum import Enum

class Entity(Enum):
    ptp_date = "ptp_date"
    time = "time"
    relation = "relation"
    person = "person"
    organisation = "organisation"
    location =  "location"
    vehicle_brand = "vehicle_brand"
    occupation = "occupation"
    qualification = "qualification"

@dataclass
class IdentifiedEntities():
    """
    Class for capturing all the entities value of NER which we might need.
    """
    date: Dict[str, str] = field(default_factory=dict)
    time: Dict[str, str] = field(default_factory=dict)
    number: Dict[str, str] = field(default_factory=dict)
    relation: Dict[str, str] = field(default_factory=dict)
    person: Dict[str, str] = field(default_factory=dict)
    location: Dict[str, str] = field(default_factory=dict)
    organisation: Dict[str, str] = field(default_factory=dict)
    vehicle_brand: Dict[str, str] = field(default_factory=dict)
    occupation: Dict[str, str] = field(default_factory=dict)
    qualification: Dict[str, str] = field(default_factory=dict)

@dataclass
class PostprocessingConfig():
    """
    Class for arguments which are customisable 
    """
    utterance: str
    labels: List[str]
    current_date: Date
    lang: str = 'hindi'
    consider_word_cnt: int = 3
    consider_due_date: bool = False 
    consider_weekend: bool=False
    back_date: bool = False
    festival_bool: bool = False


def assign_values_to_entities_class(consider_due_date: bool, ambiguos_dates:str, dates: List[Any], only_date: str) -> List[str]:
    """
    Assign the value to the date entity.

    Args:
        consider_due_date (bool): Whether we want to consider due date variable or not. 
        ambiguos_dates (str): It can be either "due_date", 'next_weekend' or "weekend". 
                                It can be extended to other types of variables dates.
        dates (List[Any]): List of all the dates which qualified the rules patterns.
        only_date (str): It is the change which we want to do in the reference dates. 
                         It will be +x or -x days.

    Returns:
        List[str]: It will return the date which needs to be returned 
    """
    if not consider_due_date and isinstance(ambiguos_dates, Date):
        if len(dates) == 0:
            return [DateHelperRefactor.convert_datetime_to_string(ambiguos_dates)]
        else:
            return [DateHelperRefactor.convert_datetime_to_string(ambiguos_dates) + timedelta(days = min(dates))]     
    if len(dates) == 0:
        return [ambiguos_dates]
    elif min(dates) < 0:
        return [ambiguos_dates + str(only_date)]
    else:
        return [ambiguos_dates + "+" + str(only_date)]



def extract_due_date(grainUnitExtraction, text: str, consider_due_date: bool, lang: str) -> str:
    """
    It extracts the due date variable date if that is present in the text. 

    Args:
        grainUnitExtraction (GrainUnitExtraction): It is GrainUnitExtraction object to get the 
        text (str): Text pattern which are labelled as date from the model.
        consider_due_date (bool): Whether we want to consider due date variable or not. 
        lang (str): Language parameter

    Returns:
        str: Return type is string. It will either be "" or "due_date"
    """
    return grainUnitExtraction.extractGrain(text, "due_date", lang) if consider_due_date else ""

def extract_weekend(grainUnitExtraction, text: str, consider_weekend: bool, lang: str) -> str:
    """
    It extracts the weekend or next weekend variable date if that is present in the text. 

    Args:
        grainUnitExtraction (GrainUnitExtraction): It is GrainUnitExtraction object to get the 
        text (str): Text pattern which are labelled as date from the model.
        consider_weekend (bool): Whether we want to consider due date variable or not. 
        lang (str): Language parameter

    Returns:
        str: Return type is string. It will either be "", "weekend" or "next_weekend"
    """
    if lang in ["hindi", "english"]:
        return grainUnitExtraction.extractGrain(text, "weekend", lang) if consider_weekend else ""
    else:
        return ""

def validate_max_length_and_filtered_entity(filtered_entity: Union[None, str, Date], max_length: int, current_max: int = -1) -> bool:
    """
    filtered_entity is not None: This condition verify that whether the date pattern is matching or not
    current_max < max_length: Verify whether new matched pattern has higher length or not.

    Args: 
        filtered_entity (Union[None, Date]): Dates after pattern matching and modification of the current_date or reference date.
                                        It can take None, str or Date type as input.
        max_length (int): Max pattern matching length.
        current_max (int): Max pattern matching length out of all the patterns matched.

    Returns:
        bool: Return True or False based on the 'and' conditions
    """
    return filtered_entity is not None and current_max < max_length

def validate_filtered_entity_is_null(filtered_entity: Union[None, str, Date]) -> bool:
    """
    filtered_entity is not None: This condition verify that whether the date pattern is matching or not.

    Args: 
        filtered_entity (Union[None, Date]): Dates after pattern matching and modification of the current_date or reference date.
                                        It can take None, str or Date type as input.

    Returns:
        bool: Return True or False
    """
    return filtered_entity is None or filtered_entity == ""

def extract_numbers(text):
    regex = r'\d+\.?\d*[kl(cr)]?'
    matches = re.findall(regex, text)
    numbers = []

    for match in matches:
        if match[-1] == 'k':
            number = int(float(match[:-1]) * 1000)
        elif match[-1] == 'l':
            number = int(float(match[:-1]) * 100000)
        elif match[-1] == 'cr':
            number = int(float(match[:-1]) * 10000000)
        else:
            number = int(float(match))
        numbers.append(number)

    return numbers

def get_number_entity(text: str) -> List[int]:
    """
    This identifies whether there is an integer in the text or not. 

    Args: 
        text (str): Text in which we want to find an integer value.

    Returns:
        List[int]: List of integers in the text. 
    """
    regex = r'\d+\.?\d*(?:\s*k|\s*l|\s*cr)?'
    matches = re.findall(regex, text)
    numbers = []

    for match in matches:
        if match[-1] == 'k':
            number = int(float(match[:-1]) * 1000)
        elif match[-1] == 'l':
            number = int(float(match[:-1]) * 100000)
        elif match[-2:] == "cr":
            number = int(float(match[:-2]) * 10000000)
        else:
            number = int(float(match))
        numbers.append(number)

    return numbers

def get_date(grainUnitExtraction: GrainUnitExtraction, postprocessingConfig: PostprocessingConfig, rulesRefactored: RulesRefactored, ptp_date: str) -> Tuple[List[Date], List[int], str, str]:
    """
    It identifies the dates which are identified from rules patterns.
    
    Args: 
        grain_variations_files (str): 
        postprocessingConfig (PostprocessingConfig): 
        rulesRefactored (RulesRefactored):
        ptp_date (str): 
    
    Returns:
        Tuple[List[Date], List[int], str, str]: 
    """
    
    dates, number = [], []
    current_max = -1
    due_date, weekend = "", ""
    for text in ptp_date:
        text = " " + text + " " 
        due_date = extract_due_date(grainUnitExtraction, text, postprocessingConfig.consider_due_date, postprocessingConfig.lang)
        weekend = extract_weekend(grainUnitExtraction, text, postprocessingConfig.consider_weekend, postprocessingConfig.lang)
        filtered_date, max_length = rulesRefactored.DateFinder(text, due_date, weekend, postprocessingConfig.current_date, \
                                                                postprocessingConfig.lang, postprocessingConfig.festival_bool, \
                                                                back_date=postprocessingConfig.back_date)

        if validate_max_length_and_filtered_entity(filtered_date, max_length, current_max):
            current_max = max_length
            dates.append(filtered_date)

        if validate_filtered_entity_is_null(filtered_date):
            number = get_number_entity(text)

    return dates, number, due_date, weekend

def get_time(grainUnitExtraction: GrainUnitExtraction, rulesRefactored: RulesRefactored, date: Date, time: str, lang: str) -> List[Date]:
    identified_time = []
    for text in time:
        text = " " + text + " "   # Adding spaces to help convert_hindi_text_to_int function to work properly
        dates = rulesRefactored.TimeFinder(text, date, lang)
        filtered_time, filtered_time_module = TimeHelper.filter_times(dates)
        if filtered_time is not None:
            identified_time.append((filtered_time, filtered_time_module))
    return identified_time

def extract_entity(rulesRefactored: RulesRefactored, text: str, lang: str, entity_type: str) -> List[str]:
    """
    Function to extract the normalized organisation names irrespective of the languages.
    """
    entity_part = rulesRefactored.entityNormalisation(entity_type, text, lang)
    return DateHelperRefactor.filter_dates(entity_part)

def get_entity(rulesRefactored: RulesRefactored, entity: str, utterance: str, lang: str, consider_word_cnt: int, entity_type: str):
    current_max = -1
    entity_list = []

    if len(entity)>0:
        for text in entity:
            text = " " + text + " "
            filtered_entity, max_length = extract_entity(rulesRefactored, text, lang, entity_type)
            if current_max < max_length:
                current_max = max_length
                if filtered_entity != None:
                    entity_list.append(filtered_entity)
        return entity_list
    if len(utterance.split()) <= consider_word_cnt:
        filtered_entity, max_length = extract_entity(rulesRefactored, utterance, lang, entity_type)
        if validate_max_length_and_filtered_entity(filtered_entity, max_length, current_max):
            current_max = max_length
            entity_list.append(filtered_entity)

    return entity_list

def get_relation(rulesRefactored: RulesRefactored, relation: str, lang: str, entity_type: str) -> List[str]:
    relation_list = []
    current_max = -1
    for text in relation:
        text = " " + text + " "
        filtered_relation, max_length = extract_entity(rulesRefactored, text, lang, entity_type)
        if validate_max_length_and_filtered_entity(filtered_relation, max_length, current_max):
            current_max = max_length
            relation_list.append(filtered_relation)

    return relation_list


# def extract_location(rulesRefactored: RulesRefactored, text: str, lang: str) -> str:
#     location_name = rulesRefactored.entityNormalisation("location", text, lang)
#     if location_name is not None:
#         return location_name
#     else: 
#         return ""

# def get_location(rulesRefactored: RulesRefactored, location: str, utterance: str, lang: str, consider_word_cnt: int) -> List[str]:
#     location_list = []
#     if len(location)>0:
#         for text in location:
#             location_value = extract_location(rulesRefactored, text, lang)
#             if len(location_value)>0:
#                 location_list.append(location_value[0])
#         return location_list
#     if len(utterance.split()) <= consider_word_cnt:
#         location_list = extract_location(rulesRefactored, utterance, lang)
#     return location_list


# def extract_organisation(rulesRefactored: RulesRefactored, text: str, lang: str) -> List[str]:
#     """
#     Function to extract the normalized organisation names irrespective of the languages.
#     """
#     organisation_part = rulesRefactored.entityNormalisation("organisation", text, lang)
#     return DateHelperRefactor.filter_dates(organisation_part)

# def get_organisation(rulesRefactored: RulesRefactored, organisation: str, utterance: str, lang: str, consider_word_cnt: int):
#     current_max = -1
#     organisation_list = []

#     if len(organisation)>0:
#         for text in organisation:
#             filtered_organisation, max_length = extract_organisation(rulesRefactored, text, lang)
#             if current_max < max_length:
#                 current_max = max_length
#                 if filtered_organisation != None:
#                     organisation_list.append(filtered_organisation)
#         return organisation_list
#     if len(utterance.split()) <= consider_word_cnt:
#         filtered_organisation, max_length = extract_organisation(rulesRefactored, utterance, lang)
#         if validate_max_length_and_filtered_entity(filtered_organisation, max_length, current_max):
#             current_max = max_length
#             organisation_list.append(filtered_organisation)

#     return organisation_list


# def extract_vehicle_brand(rulesRefactored: RulesRefactored, text: str, lang: str):
#     """
#     Extract 
#     """
#     vehicle_brand_part = rulesRefactored.entityNormalisation("vehicle_brand", text, lang)
#     return DateHelperRefactor.filter_dates(vehicle_brand_part)

# def get_vehicle_brand(rulesRefactored: RulesRefactored, vehicle_brand: str, utterance: str, lang: str, consider_word_cnt: int):
#     current_max = -1
#     vehicle_brand_list = []

#     if len(vehicle_brand)>0:
#         for text in vehicle_brand:
#             filtered_vehicle_brand, max_length = extract_vehicle_brand(rulesRefactored, text, lang)
#             if current_max < max_length:
#                 current_max = max_length
#                 if filtered_vehicle_brand != None:
#                     vehicle_brand_list.append(filtered_vehicle_brand)
#         return vehicle_brand_list
#     if len(utterance.split()) <= consider_word_cnt:
#         filtered_vehicle_brand, max_length = extract_vehicle_brand(rulesRefactored, utterance, lang)
#         if validate_max_length_and_filtered_entity(filtered_vehicle_brand, max_length, current_max):
#             current_max = max_length
#             vehicle_brand_list.append(filtered_vehicle_brand)

#     return vehicle_brand_list

# def extract_occupation(rulesRefactored: RulesRefactored, text: str, lang: str) -> List[str]:
#     """
#     Function to extract the normalized occupation names irrespective of the languages.
#     """
#     occupation_part = rulesRefactored.entityNormalisation("occupation", text, lang)
#     return DateHelperRefactor.filter_dates(occupation_part)

# def get_occupation(rulesRefactored: RulesRefactored, occupation: str, utterance: str, lang: str, consider_word_cnt: int):
#     current_max = -1
#     occupation_list = []

#     if len(occupation)>0:
#         for text in occupation:
#             filtered_occupation, max_length = extract_occupation(rulesRefactored, text, lang)
#             if current_max < max_length:
#                 current_max = max_length
#                 if filtered_occupation != None:
#                     occupation_list.append(filtered_occupation)
#         return occupation_list
#     if len(utterance.split()) <= consider_word_cnt:
#         filtered_occupation, max_length = extract_occupation(rulesRefactored, utterance, lang)
#         if validate_max_length_and_filtered_entity(filtered_occupation, max_length, current_max):
#             current_max = max_length
#             occupation_list.append(filtered_occupation)

#     return occupation_list

# def extract_qualification(rulesRefactored: RulesRefactored, text: str, lang: str) -> List[str]:
#     """
#     Function to extract the normalized organisation names irrespective of the languages.
#     """
#     qualification_part = rulesRefactored.entityNormalisation("qualification", text, lang)
#     return DateHelperRefactor.filter_dates(qualification_part)

# def get_qualification(rulesRefactored: RulesRefactored, qualification: str, utterance: str, lang: str, consider_word_cnt: int):
#     current_max = -1
#     qualification_list = []

#     if len(qualification)>0:
#         for text in qualification:
#             filtered_qualification, max_length = extract_qualification(rulesRefactored, text, lang)
#             if current_max < max_length:
#                 current_max = max_length
#                 if filtered_qualification != None:
#                     qualification_list.append(filtered_qualification)
#         return qualification_list
#     if len(utterance.split()) <= consider_word_cnt:
#         filtered_qualification, max_length = extract_qualification(rulesRefactored, utterance, lang)
#         if validate_max_length_and_filtered_entity(filtered_qualification, max_length, current_max):
#             current_max = max_length
#             qualification_list.append(filtered_qualification)

#     return qualification_list

