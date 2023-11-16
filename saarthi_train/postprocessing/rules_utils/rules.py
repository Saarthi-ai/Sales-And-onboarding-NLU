import re
from .. import utils
from .rules_pattern import Pattern
from typing import Union, List, Dict
from datetime import datetime as Date
from ..OffsetHelper import GrainUnitExtraction
from ..time_utils import TimeHelper, DateHelperRefactor, AmountHelper


def check_variable_date_condition(date_diff: Union[int, str], festival_bool: bool) -> bool:
    """
    Check if a given date difference meets certain conditions.
    These conditions identify whether offset is able to find any date pattern or not.

    Args:
        date_diff (Union[int, str]): The difference in days between two dates.
            Can be an int or the string "none".
        festival_bool (bool): Indicates whether or not a festival is happening
            on the given date.

    Returns:
        bool: True if the conditions are met, False otherwise.
    """
    return date_diff!="none" and date_diff < 0 and festival_bool == False

def variable_type_checker(due_date: str, weekend: str) -> bool:
    """
    Check if a given date difference meets certain conditions.
    These conditions identify whether offset is able to find any date pattern or not.

    Args:
        date_diff (Union[int, str]): The difference in days between two dates.
            Can be an int or the string "none".
        festival_bool (bool): Indicates whether or not a festival is happening
            on the given date.

    Returns:
        bool: True if the conditions are met, False otherwise.
    """
    return due_date == "due_date" or weekend == "weekend" or weekend == "next_weekend"

class RulesRefactored:

    def __init__(self, lang: str, grainUnitExtraction: GrainUnitExtraction = None):
        self.pattern = Pattern()
        self.grainUnitExtraction = grainUnitExtraction
        self.lang = lang

    def get_pattern_based_on_entity_type(self, entity: str) -> Dict[str, List[str]]:
        if entity == "relation":
            return self.pattern.get_relations(self.lang)
        if entity == "location":
            return self.pattern.get_location(self.lang)
        if entity == "vehicle_brand":
            return self.pattern.get_vehicle_brand(self.lang)
        if entity == "organisation":
            return self.pattern.get_organisations(self.lang)
        if entity == "vehicle_brand":
            return self.pattern.get_vehicle_brand(self.lang)
        if entity == "occupation":
            return self.pattern.get_occupation(self.lang)
        if entity == "qualification":
            return self.pattern.get_qualification(self.lang)
        return {}

    def DateFinder(self, text: str, due_date: str, weekend: str, date: Date, lang: str = "hindi", festival_bool: bool = False, back_date: bool=False):
        dates = []
        patterns_dict = self.pattern.get_pattern_date(lang)
        for sequence in patterns_dict.values():
            pattern, offset_sequence, grain = sequence[0], sequence[1], sequence[2]
            pattern_text = re.findall(pattern, text)
            if len(pattern_text) == 0:
                continue
            dateHelper = DateHelperRefactor(grain, back_date=back_date, grainUnitExtraction = self.grainUnitExtraction)
            match_length = utils.get_match_length(pattern_text, lang)
            dateHelper.update_offset(pattern_text, offset_sequence, lang)
            updatedDate, date_diff = dateHelper.update(date)
            if variable_type_checker(due_date, weekend) or check_variable_date_condition(date_diff, festival_bool):
                dates.append((date_diff, match_length))
            else:
                dates.append((updatedDate, match_length))
        filtered_date, max_length = DateHelperRefactor.filter_dates(dates)
        if festival_bool == True and filtered_date == None:
            return date, 0
        return filtered_date, max_length

    def TimeFinder(self, text: str, date: Date, lang: str = "hindi"):
        dates = []
        current_max, time_module = -1, "later"
        patterns_dict = self.pattern.get_pattern_time(lang)
        for sequence in patterns_dict.values():
            pattern, offset_sequence, grain = sequence[0], sequence[1], sequence[2]
            pattern_text = re.findall(pattern, text)
            if len(pattern_text) == 0:
                continue
            timeHelper = TimeHelper(grain, grainUnitExtraction = self.grainUnitExtraction)
            match_length = utils.get_match_length(pattern_text, lang)
            timeHelper.update_offset(pattern_text, offset_sequence, lang)
            if current_max <= match_length:
                current_max = match_length
                updated_time, time_module = timeHelper.update(date)
                dates.append((updated_time, time_module))
        return dates

    def amountFinder(self, text: str, lang: str = "hindi"):
        amount = []
        patterns_dict = self.pattern.get_pattern_amount(lang)
        for sequence in patterns_dict.values():
            pattern, offset_sequence, grain = sequence[0], sequence[1], sequence[2]
            pattern_text = re.findall(pattern, text)
            if len(pattern_text) == 0:
                amount.append("No pattern found")
                continue
            amountHelper = AmountHelper(grain)
            amountHelper.update_offset(pattern_text, offset_sequence, lang)
            amount.append(amountHelper.update())
        return amount

    def entityNormalisation(self, entity_type, text: str, lang: str):
        entity = []
        patterns_dict = self.get_pattern_based_on_entity_type(entity_type)
        for sequence in patterns_dict.values():
            pattern, normalized_entity = sequence[0], sequence[1]
            pattern_text = re.findall(pattern, text)
            if len(pattern_text) == 0:
                continue
            match_length = utils.get_match_length(pattern_text, lang)
            entity.append((normalized_entity, match_length))
        return entity

    # def relationNormalization(self, text: str, lang: str = "hindi"):
    #     relation = []
    #     patterns_dict = self.pattern.get_relations(lang)
    #     for sequence in patterns_dict.values():
    #         pattern, normalized_relation = sequence[0], sequence[1]
    #         pattern_text = re.findall(pattern, text)
    #         if len(pattern_text) == 0:
    #             continue
    #         match_length = utils.get_match_length(pattern_text, lang)
    #         relation.append((normalized_relation, match_length))
    #     return relation

    # def organisationNormalization(self, text: str, lang: str = "hindi"):
    #     organisation = []
    #     patterns_dict = self.pattern.get_organisations(lang)
    #     for sequence in patterns_dict.values():
    #         pattern, normalized_organisation = sequence[0], sequence[1]
    #         pattern_text = re.findall(pattern, text)
    #         if len(pattern_text) == 0:
    #             continue
    #         match_length = utils.get_match_length(pattern_text, lang)
    #         organisation.append((normalized_organisation, match_length))
    #     return organisation

    # def vehicleBrandNormalization(self, text: str, lang:str = "hindi"):
    #     vehicle_brand = []
    #     patterns_dict = self.pattern.get_vehicle_brand(lang)
    #     for sequence in patterns_dict.values():
    #         pattern, normalized_vehicle_brand = sequence[0], sequence[1]
    #         pattern_text = re.findall(pattern, text)
    #         if len(pattern_text) == 0:
    #             continue
    #         vehicle_brand.append(normalized_vehicle_brand)
    #     if len(vehicle_brand)>0:
    #         return vehicle_brand[0]
    #     else:
    #         return ""

    # def locationNormalization(self, text, lang = "hindi"):
    #     location = []
    #     patterns_dict = self.pattern.get_location(lang)
    #     for sequence in patterns_dict.values():
    #         pattern, normalized_location = sequence[0], sequence[1]
    #         pattern_text = re.findall(pattern, text)
    #         if len(pattern_text) == 0:
    #             continue
    #         location.append(normalized_location)
    #     if len(location)>0:
    #         return location[0]
    #     else:
    #         return ""