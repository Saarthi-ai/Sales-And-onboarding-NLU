from calendar import SATURDAY, month, week
import re, json
from typing import Union, List

class GrainUnitExtraction:

    def __init__(self, grain_variants_file: str):
        with open(grain_variants_file) as f:
            self.grain_variants = json.loads(f.read())
    
    def extractIntPos(self, text: str, i: int):
        try:
            text = " ".join(text[0]).strip()
            temp = re.findall(r'\b\d+\b', text)
            temp = list(map(int, temp))
            return temp[i]
        except Exception as ex:
            return -1

    def extractInt(self, text: str):
        text = " ".join(text[0]).strip()
        temp = re.findall(r'\b\d{1,2}\b', text)
        if len(temp) == 0:
            return -1
        temp = list(map(int, temp))
        return temp[0]

    def extractYear(self, text: str):
        text = " ".join(text[0]).strip()
        temp = re.findall(r'\b\d{4}\b', text)
        if len(temp) == 0:
            return -1
        temp = list(map(int, temp))
        return temp[0]

    def extractFloat(self, text: str):
        text = " ".join(text[0]).strip()
        temp = re.findall(r'[0-9,.]+', text)
        if len(temp) == 0:
            return -1
        temp = list(map(float, temp))
        return temp[0]
    
    def extractGrain(self, text: Union[str, List[str]], grain: str, lang: str) -> Union[str, int]:
        """
        Extracts the grain value based on arguments.

        Args:
            text (str): Text in which we want to find any grain value
            Grain (str): Pass grain as one of these values - 
                        ["weekend", "month", "weekday", "meridiam", "daytime", "time_module", "due_date"]
                        
            lang (str):  Language parameter
        
        Returns:
            str | int: It returns either string or integer based on what type of grain is passed as a parameter
        """
        # TODO validate it as enum which is to be taken from grain units file itself.
        str_grain = ['weekend', 'due_date']
        # if grain == "daytime":
        #     text = text[0]
        if isinstance(text, str):
            text = text
        elif isinstance(text, list) and all(isinstance(item, str) for item in text):
            text = text[0]
        elif isinstance(text, list) and all(isinstance(item, tuple) for item in text):
        # elif grain in ["daytime", "month", "weekday", "time_module"]:
            text = " ".join(text[0])
        else:
            text = " ".join(text[0])
        for grain_keys, grain_values in self.grain_variants[lang][grain].items():
            pattern = re.search(grain_values, text)
            if pattern != None and pattern.group() != "":
                if grain in str_grain:
                    return grain_keys
                return int(grain_keys)
        return ""

    def extractAmount(self, text: str, lang: str):
        """
            TODO: Integrate amount function to return the amount exact value
        """
        return None
