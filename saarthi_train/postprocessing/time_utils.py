import pytz
import logging
import calendar
import datetime as Date
from datetime import timedelta
from .OffsetHelper import GrainUnitExtraction
from typing import Any, List, Tuple

class DateHelperRefactor:
    
    def __init__(self, grain=None, offset={}, back_date = False, grainUnitExtraction: GrainUnitExtraction = None):
        self.offset = offset
        self.grain = grain
        self.back_date = back_date
        self.grainUnitExtraction = grainUnitExtraction
        
    def validate_offset(self):
        if self.grain == "Day":
            if self.offset[0] <= 31 and self.offset[0]>=1:
                return True
            else:
                return False
    
    @staticmethod
    def convert_datetime_to_string(date):
        # logging.info(f"Convert datetime to string: {date}")
        datestring = date.strftime("%d/%m/%Y")
        # logging.info(f'Datestring: {datestring}')
        return datestring
    
    def validate_dates(self, day, month, year):
        return day.replace(day = calendar.monthrange(year, month)[1], month = month, year = year)
    
    def update(self, time_to_update):
        if self.grain == "Day":
            try:
                mod = self.offset.get("mod", -1) if "mod" in self.offset or self.back_date else 1
                if self.offset is not None and "int" in self.offset and isinstance(self.offset["int"], int):
                    days_to_add = self.offset["int"] * mod
                    return time_to_update + timedelta(days=days_to_add), days_to_add
                if self.offset is not None and "future_mod" in self.offset and isinstance(self.offset["future_mod"], int):
                    days_to_add = self.offset["future_mod"] * mod
                    return time_to_update + timedelta(days=days_to_add), days_to_add
            except ValueError as err:
                month = time_to_update.month
                year = time_to_update.year
                return (self.validate_dates(time_to_update, month, year) if self.back_date else 
                        self.validate_dates(time_to_update, month, year-1)), self.offset["int"]*mod
        elif self.grain == "Date":
            year, month, date = time_to_update.year, time_to_update.month, time_to_update.day
            if self.offset is not None and "year" in self.offset and isinstance(self.offset["year"], int):
                year = self.offset["year"]
            if self.offset is not None and "month" in self.offset and isinstance(self.offset["month"], int):
                month = self.offset["month"]
            if self.offset is not None and "int" in self.offset and isinstance(self.offset["int"], int):
                date = self.offset["int"]
            elif self.offset is not None and "int" not in self.offset:
                date = calendar.monthrange(year, month)[1] if self.offset is not None and "end" in self.offset and self.offset["end"] == 1 else 1
            if "int" not in self.offset and month == time_to_update.month and not self.back_date:
                date = calendar.monthrange(year, month)[1]
            if date < time_to_update.day and month == time_to_update.month and "month" not in self.offset:
                month = month + 1 if month < 12 else 1
                year = year + 1 if month == 1 else year
            try:
                response = time_to_update.replace(day=date, month=month, year=year)
            except ValueError as err:
                response = self.validate_dates(time_to_update, month, year)
            if response < time_to_update:
                response = response.replace(year=year) if self.back_date or "year" in self.offset else response.replace(year=year+1)
            else:
                if "month" in self.offset:
                    response = response.replace(year=year-1) if self.back_date else response.replace(year=year)
                else:
                    if self.back_date:
                        if month>1:
                            response = response.replace(month = month-1)  
                        else:
                            response = response.replace(month=12, year=year-1)     
                    else:
                        response = response.replace(year=year)     
            return response, "none"
        elif self.grain == "Month":
            month, year = time_to_update.month, time_to_update.year
            
            if "mod" in self.offset or self.back_date:
                mod = self.offset.get("mod", -1)
            else:
                mod = 1
                
            if "future_mod" in self.offset:
                future_mod = self.offset.get("future_mod")
                if isinstance(future_mod, int):
                    month += future_mod * mod
            elif "int" in self.offset:
                offset_int = self.offset.get("int")
                if isinstance(offset_int, int):
                    month += offset_int * mod
                    
            year += (month - 1) // 12
            month = (month - 1) % 12 + 1
            
            try:
                day = self.offset.get("int", 1) if "future_mod" not in self.offset else self.offset.get("int")
                if "end" in self.offset:
                    day = calendar.monthrange(year, month)[1]
                else:
                    day = 1
                # if response < time_to_update:
                #     response = response.replace(year=year) if self.back_date or "year" in self.offset else response.replace(year=year+1)
                # else:
                #     if "month" in self.offset:
                #         response = response.replace(year=year-1) if self.back_date else response.replace(year=year)
                #     else:
                #         if self.back_date:
                #             if month>1:
                #                 response = response.replace(month = month-1)  
                #             else:
                #                 response = response.replace(month=12, year=year-1)     
                #         else:
                #             response = response.replace(year=year)  
                return time_to_update.replace(day=day, month=month, year=year), "none"
                # return response, "none"
            except Exception as ex:
                return time_to_update.replace(day=calendar.monthrange(year, month)[1], month=month, year=year), "none"

        # elif self.grain == "Month":
        #     date = time_to_update
        #     month = date.month
        #     if "mod" in self.offset or self.back_date:
        #         try:
        #             mod = self.offset["mod"]
        #         except Exception as ex:
        #             mod = -1
        #     else:
        #         mod = 1
        #     if "future_mod" in self.offset:
        #         if type(self.offset["future_mod"]) == int:
        #             month = date.month + self.offset["future_mod"]*mod
        #     if "future_mod" not in self.offset and "int" in self.offset:
        #         if type(self.offset["int"]) == int:
        #             month = date.month + self.offset["int"]*mod
        #     year = time_to_update.year
        #     if month > 12:
        #         year = year + (month)//12
        #         if month%12 != 0:
        #             month = month%12
        #         else:
        #             month = 12
        #     try:
        #         if "int" in self.offset and "future_mod" in self.offset:
        #             if type(self.offset["int"]) == int:                
        #                 day = self.offset["int"]
        #         else:
        #             if "end" in self.offset:
        #                 day = calendar.monthrange(date.year, month)[1]
        #             else:
        #                 day = 1
        #         return date.replace(day = day, month = month, year = year), "none"
        #     except Exception as ex:
        #         return date.replace(day = calendar.monthrange(date.year, month)[1], month = month, year = year), "none"
        elif self.grain == "Week":
            date = time_to_update
            current_week = date.isoweekday()
            mod = self.offset.get("mod", -1) if "mod" in self.offset or self.back_date else 1
            if "end" in self.offset:
                if current_week == 7:
                    day_diff = 7 * (self.offset.get("int", 0) + 1) * mod if "int" in self.offset else 7 * self.offset.get("future_mod", 0) * mod
                else:
                    day_diff = (7 * (self.offset.get("int", 0) + 1) - current_week) * mod if "int" in self.offset else (7 * self.offset.get("future_mod", 0) - current_week) * mod
            elif "weekday" in self.offset:
                day_diff = (7 * self.offset.get("future_mod", 0) - (current_week - self.offset["weekday"])) if current_week >= self.offset["weekday"] and "future_mod" in self.offset else (7 * self.offset.get("int", 0) - (current_week - self.offset["weekday"])) if "int" in self.offset else (self.offset["weekday"] - current_week)
            else:
                day_diff = 7 * self.offset.get("int", 0) * mod if "int" in self.offset else 7 * self.offset.get("future_mod", 0) * mod
            if self.back_date:
                return date + (timedelta(days=day_diff - 7)), 7 - day_diff
            else:
                return date + timedelta(days=day_diff), day_diff
        elif self.grain == "Year":
            mod = -1 if self.back_date else 1
            if "future_mod" in self.offset and isinstance(self.offset["future_mod"], int):
                day_to_increase = 365*self.offset["future_mod"]*mod
            if "int" in self.offset and isinstance(self.offset["int"], int):
                day_to_increase = 365*int(self.offset["int"])*mod
            return time_to_update + timedelta(days = day_to_increase), day_to_increase*mod
        elif self.grain == "MonthWeek":  # ["future_mod1"->for month, "int", "weekday", "mod"]
            date = time_to_update
            month = date.month
            # if "mod" in self.offset and self.back_date:
            #     mod = self.offset["mod"]
            # else:
            #     mod = 1
            if "future_mod" in self.offset:
                if type(self.offset["future_mod"]) == int:
                    month = date.month + self.offset["future_mod"]
            year = time_to_update.year
            if month > 12:
                year = year + (month)//12
                if month%12 != 0:
                    month = month%12
                else:
                    month = 12
            try:
                day = 1
                date = date.replace(day = day, month = month, year = year)
                current_week = date.isoweekday()
                if current_week >= self.offset["weekday"]: 
                    if "int" in self.offset:
                        day_diff = 7*self.offset["int"] - (current_week - self.offset["weekday"])
                    else:
                        day_diff = 7 - (current_week - self.offset["weekday"])
                else:
                    day_diff = self.offset["weekday"] - current_week
                return date + timedelta(days=day_diff), "none"
            except Exception as ex:
                return date.replace(day = 1, month = month, year = year), "none"

    def update_offset(self, text, sequence_list, lang):
        try:
            offset = {}
            if "int" in sequence_list:
                int_extracted = self.grainUnitExtraction.extractInt(text)
                if int_extracted != -1:
                    offset["int"] = int_extracted  
            if "month" in sequence_list:
                month_extracted = self.grainUnitExtraction.extractGrain(text, "month", lang)
                if month_extracted != "":
                    offset["month"] = month_extracted  
            if "year" in sequence_list:
                int_extracted = self.grainUnitExtraction.extractYear(text)
                if int_extracted != -1:
                    offset["year"] = int_extracted  
            if "weekday" in sequence_list:
                weekday_extracted = self.grainUnitExtraction.extractGrain(text, "weekday", lang)
                if weekday_extracted != "":
                    offset["weekday"] = weekday_extracted
            if "future_mod0" in sequence_list:
                future_mod = self.grainUnitExtraction.extractInt(["future_mod0"])
                if future_mod != -1:
                    offset["future_mod"] = future_mod
            if "future_mod1" in sequence_list:
                future_mod = self.grainUnitExtraction.extractInt(["future_mod1"])
                if future_mod != -1:
                    offset["future_mod"] = future_mod
            if "future_mod2" in sequence_list:
                future_mod = self.grainUnitExtraction.extractInt(["future_mod2"])
                if future_mod != -1:
                    offset["future_mod"] = future_mod
            if "mod" in sequence_list:
                mod = self.grainUnitExtraction.extractGrain(text, "time_module", lang)
                if mod != "":
                    offset["mod"] = mod
            if "end" in sequence_list:
                offset["end"] = 1
            self.offset = offset
            return offset 
        except Exception as ex:
            # logger.error(f"Unresolved offset: {ex.args}")
            return -1

    @staticmethod
    def filter_dates(dates_list):
        valid_list = []
        for dates in dates_list:
            if dates[0] != 'No pattern found' and dates[0] != "Invalid date":
                valid_list.append(dates)

        m = -1000000000
        response = []

        for dates in valid_list:
            if dates[1] > m:
                m = dates[1]
                response = [dates[0]]
            elif dates[1] == m:
                m = dates[1]
                response.append(dates[0])

        if len(response) == 0:
            return None, -1
        return min(response), m


class TimeHelper:
    
    def __init__(self, grain, offset = {}, grainUnitExtraction: GrainUnitExtraction = None):
        self.offset = offset
        self.grain = grain
        self.grainUnitExtraction = grainUnitExtraction
    
    @staticmethod
    def convert_datetime_to_string(date):
        return date.strftime("%H:%M")
    
    @staticmethod
    def validate_time(date: Date, text: str, grainUnitExtraction: GrainUnitExtraction, lang : str = "hindi"):
        '''
            If there is day_time or meridiam given like evening, afternoon or night -> add 12 hours
            Conditions:
                Predicted hour should be less than 12. 
        '''

        hour = date.hour
        current_date_time = Date.datetime.now(pytz.timezone('Asia/Kolkata'))
        current_hour, current_date = current_date_time.hour, current_date_time.day
        meridiam = grainUnitExtraction.extractGrain(text, "meridiam", lang)
        daytime = grainUnitExtraction.extractGrain(text, "daytime", lang)
        if (daytime in [12, 18, 20] or meridiam == 2 or (current_hour > hour and current_date == date.day and hour != 12)) and hour != daytime and hour <= 12:
            hour += 12
            if daytime == 20 and daytime >= 1 and daytime <= 4:
                hour -= 12
            if hour >= 24:
                if hour == 24:
                    hour = 0
                else:
                    hour = 23
        date = date.replace(hour = hour)
        if date < Date.datetime.now(pytz.timezone('Asia/Kolkata')): 
            date = date + timedelta(days=1)
        return date

    
    def validate_dates(self, day, month, year):
        return day.replace(day = calendar.monthrange(year, month)[1], month = month, year = year)
    
    def update(self, time_to_update):
        response = time_to_update
        if self.grain == "relative_time":
            if "hour" in self.offset and self.offset["hour"] != -1:
                try:
                    response = response + timedelta(hours = self.offset["hour"])
                except Exception as ex:
                    # logger.error(f"Unresolved hour: {ex.args}")
                    pass
            return response, self.offset["mod"]
        elif self.grain == "relative_minute":
            if "minute" in self.offset and self.offset["minute"] != -1:
                try:
                    response = response + timedelta(minutes = self.offset["minute"])
                except Exception as ex:
                    # logger.error(f"Unresolved minute: {ex.args}")
                    pass
            return response, self.offset["mod"]
        elif self.grain == "exact_time":
            if self.offset["meridiam"] != -1:
                response = response
            if "minute" in self.offset and self.offset["minute"]!=-1:
                try:
                   response = response.replace(minute = self.offset["minute"])
                except Exception as ex:
                    # logger.error(f"Unresolved minute: {ex.args}")
                    pass
            else:
                response = response.replace(minute = 0)
            if "hour" in self.offset and self.offset["hour"] != -1:
                try:
                    response = response.replace(hour = self.offset["hour"])
                    # response = response.replace(minute = 0)
                except Exception as ex:
                    # logger.error(f"Unresolved hour: {ex.args}")
                    pass
            return response, self.offset["mod"]
        elif self.grain == "exact_minute":
            try:
                return response + timedelta(minutes = self.offset["minute"]), self.offset["mod"]
            except Exception as ex:
                # logger.error(f"Unresolved minute: {ex.args}")
                pass
        elif self.grain == "daytime":
            try:
                if self.offset["daytime"] == -1:
                    return response, self.offset["mod"]
                return response.replace(hour = self.offset["daytime"], minute = 0), self.offset["mod"]
            except Exception as ex:
                # logger.error(f"Unresolved daytime: {ex.args}")
                pass

    @staticmethod
    def filter_times(dates_list: List[Any]):
        valid_list = []
        for dates in dates_list:
            if dates[0] != 'No pattern found' and dates[0] != "Invalid date" and dates[0] != "":
                valid_list.append(dates)
        if len(valid_list) == 0:
            return None, None

        # Initialize minimum date and value
        min_date = valid_list[0][0]
        min_value = valid_list[0][1]

        # Loop through the list of tuples
        for date, value in valid_list:
            # Update the minimum date and value if necessary
            if date < min_date:
                min_date = date
                min_value = value
        return min_date, min_value

    def update_offset(self, text, sequence_list, lang):
        try:
            offset = {}
            if "hour" in sequence_list:
                hour = self.grainUnitExtraction.extractInt(text)
                if hour == -1:
                    offset["hour"] = 1
                else:
                    offset["hour"] = hour
            if "meridiam" in sequence_list:
                text = " ".join(text[0])
                offset["meridiam"] = self.grainUnitExtraction.extractGrain(text, "meridiam", lang)
            if "minute" in sequence_list:
                offset["minute"] = self.grainUnitExtraction.extractIntPos(text, 0)
            if "hour" in sequence_list and "minute" in sequence_list:
                offset["minute"] = self.grainUnitExtraction.extractIntPos(text, 1)
            if "daytime" in sequence_list:
                offset["daytime"] = self.grainUnitExtraction.extractGrain(text, "daytime", lang)
            if "mod" in sequence_list:
                offset["mod"] = self.grainUnitExtraction.extractGrain(text, "time_module", lang)
            self.offset = offset
            return offset 
        except Exception as ex:
            # logger.error(f"Unresolved offset: {ex.args}")
            return -1
        

class AmountHelper:
    
    def __init__(self, grain: str, grainUnitExtraction: GrainUnitExtraction,  offset: str = None,):
        self.offset = offset
        self.grain = grain
        self.grainUnitExtraction = grainUnitExtraction

    def update(self):
        # response = time_to_update
        if self.grain == "exactAmount":
            try:
                response = self.offset["int"]
                return response
            except Exception as ex:
                pass
        if self.grain == "crore":
            try:
                response = self.offset["int"]*self.offset["crore"]
                return response
            except Exception as ex:
                pass
        if self.grain == "lakh":
            try:
                response = self.offset["int"]*self.offset["lakh"]
                return response
            except Exception as ex:
                pass
        if self.grain == "thousand":
            try:
                response = self.offset["int"]*self.offset["thousand"]
                return response
            except Exception as ex:
                pass

    @staticmethod
    def filter_amount(dates_list):
        valid_list = []
        for dates in dates_list:
            if dates != 'No pattern found' and dates != "Invalid date":
                valid_list.append(dates)
        if len(valid_list) == 0:
            return None
        return min(valid_list)

    def update_offset(self, text, sequence_list, lang):
        try:
            offset = {}
            if "int" in sequence_list:
                # idx = sequence_list.index("exact_time")
                amount = self.grainUnitExtraction.extractFloat(text)
                if amount == -1:
                    offset["int"] = 1
                else:
                    offset["int"] = amount
            if "crore" in sequence_list:
                offset["crore"] = self.grainUnitExtraction.extractAmount(text)
            if "lakh" in sequence_list:
                # idx = sequence_list.index("exact_minute")
                offset["lakh"] = self.grainUnitExtraction.extractAmount(text)
            if "thousand" in sequence_list:
                offset["thousand"] = self.grainUnitExtraction.extractAmount(text)
            self.offset = offset
            return offset 
        except Exception as ex:
            return -1
        