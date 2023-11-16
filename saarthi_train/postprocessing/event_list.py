import re
from datetime import datetime as Date
import pytz
# import holidays

def event_list(lang = "english"):

    if lang == "hindi":
        return {
            "New Year\'s Day": ["नव वर्ष|नए साल|नया साल|न्यू ईयर", "01/01"],
            "Pongal": ["पोंगल|पोगल", "14/01"], 
            "Makar Sankranti":["मकर संक्रांति|मकर संक्रांत", "14/01"],
            "Republic Day":	["रिपब्लिक डे|गणतंत्र दिवस", "26/01"],
            "Valentine day": ["वैलेंटाइंस डे", "14/02"],
            "Maha Shivaratri/Shivaratri": ["महाशिवरात्रि|महा शिवरात्रि|शिवरात्रि","01/03"],
            "Holi":	["होली", "08/03"],
            "Ugadi": ["उगादि", "02/04"],
            "Ramzan Id/Eid-ul-Fitar": ["रमजान ईद","03/05"], 
            "Bakr Id/Eid ul-Adha": ["बकरीद","10/07"], 
            "Muharram/Ashura": ["मुहर्रम|असुरा|आशूरा", "09/08"],
            "Raksha Bandhan": ["रक्षाबंधन|राखी","11/08"],
            "Varamahalakshmi": ["वरामहालक्ष्मी|वरा महालक्ष्मी|बरा महालक्ष्मी|बरामहालक्ष्मी","12/08"],
            "Independence Day":	["इंडिपेंडेंस डे|स्वतंत्रता दिवस","15/08"],
            "Janmashtami":	["जन्माष्टमी","19/08"],
            "Mahatma Gandhi Jayanti": ["महात्मा गांधी जयंती|गांधी जयंती", "02/09"], 
            "Ganesh Chaturthi":	["गणेश चतुर्थी","31/09"],
            "Onam":	["पूनम|ओणम","08/09"],
            "First Day of Sharad Navratri":	["शरद नवरात्रि","26/09"],	
            "Dussehra,Vijaya dashami,Dasara": ["दशहरा|विजयदशमी|विजय दशमी|दशहरा", "05/10"],
            "Diwali/Deepavali":	["दिवाली|दीपावली", "24/10"],
            "Christmas":	["क्रिसमस", "25/12"],
            "Gudi Padwa": ["गुड़ी पड़वा", "02/04"],
            "Shri Rama Navami": ["श्री राम नवमी|राम नवमी|रामनवमी", "30/03"],
            "Holi Ekadashi": ["होली एकादशी|होलिका दहन", "07/03"], 
            "Vaisakhi": ["वैशाखी", "14/04"],
            "Good Friday": ["गुड फ्राइडे", "15/02"],
            "Easter Day": ["ईस्टर", "17/04"],
            "First Day of Durga Puja Festivities": ["दुर्गा पूजा", "01/10"],
            "Maha Saptami": ["महा सप्तमी", "02/10"],
            "Maha Ashtami": ["महा अष्टमी|महाअष्टमी", "03/10"],
            "Maha Navami": ["महानवमी|महा नवमी", "04/10"],
            "Bhai Duj": ["भाई दूज|भाईदूज", "26/10"],
            "Chhat Puja" : ["छठ पूजा", "30/10"],
            "Guru Nanak Jayanti": ["गुरु नानक जयंती|नानक जयंती", "29/12"],
            "Christmas Eve": ["क्रिसमस इव", "24/12"],
            "New years eve": ["न्यू ईयर्स इव", "31/12"],
            "Halloween": ["हेलो इन|हेलोवीन", "31/10"],
            "Friendship day": ["फ्रेंडशिप डे", "07/08"],
            "Mothers day": ["मदर्स डे", "08/05"],
            "Fathers day": ["फादर्स डे", "19/06"],
        }
    
    if lang == "english":
        return {
            "New Year\'s Day": ["new year\'s day|new year day|new year", "01/01"],
            "Pongal": ["pongal", "14/01"], 
            "Makar Sankranti":["makar sankranti|makar sankrant", "14/01"],
            "Republic Day":	["republic day", "26/01"],
            "Maha Shivaratri/Shivaratri": ["maha shivaratri|shivarathi|shivratri","01/03"],
            "Holi":	["holi", "08/03"],
            "Ugadi": ["ugadi", "02/04"],
            "Ramzan Id/Eid-ul-Fitar": ["ramzan id|eid ul fitar","03/05"], 
            "Bakr Id/Eid ul-Adha": ["bakr id|eid ul-adha","10/07"], 
            "Muharram/Ashura": ["muharram|ashura", "09/08"],
            "Raksha Bandhan": ["raksha bandhan|rakhi","11/08"],
            "Varamahalakshmi": ["varamahalakshmi","12/08"],
            "Independence Day":	["independence day|happy independence day","15/08"],
            "Janmashtami":	["janmashtami","19/08"],
            "Mahatma Gandhi Jayanti": ["mahatma gandhi jayanti|gandhi jayanti", "02/09"], 
            "Ganesh Chaturthi":	["ganesh chaturthi","31/09"],
            "Onam":	["onam","08/09"],
            "First Day of Sharad Navratri":	["sharad navrati","26/09"],	
            "Dussehra,Vijaya dashami,Dasara": ["dasara|dussehra|vijaya dashami|dasara", "05/10"],
            "Diwali/Deepavali":	["diwali|deepavali", "24/10"],
            "Christmas":	["christmas", "25/12"],
            "Gudi Padwa": ["gudi padwa", "02/04"],
            "Shri Rama Navami": ["shree rama navami", "30/03"],
            "Holi Ekadashi": ["holi ekadashi|holika dahan|holika dehan", "07/03"], 
            "Vaisakhi": ["vaisakhi", "14/04"],
            "Good Friday": ["good friday", "15/04"],
            "Easter Day": ["easter day", "17/04"],
            "First Day of Durga Puja Festivities": ["durja puja", "01/10"],
            "Maha Saptami": ["maha saptami", "02/10"],
            "Maha Ashtami": ["maha ashtami", "03/10"],
            "Maha Navami": ["maha navami", "04/10"],
            "Bhai Duj": ["bhai duj|bhai dooj", "26/10"],
            "Chhat Puja" : ["chhat puja|chhat pooja", "30/10"],
            "Guru Nanak Jayanti": ["guru nanak jayanti", "29/12"],
            "Christmas Eve": ["x mas|christmas eve|christmas evening", "24/12"],
            "New years eve": ["new year eve|new years eve", "31/12"],
            "Halloween": ["halloween", "31/10"],
            "Friendship day": ["friendship day", "07/08"],
            "Mothers day": ["mothers day", "08/05"],
            "Fathers day": ["fathers day", "19/06"],
        }

    if lang == "kannada":
        return {
            "New Year\'s Day": ["ಹೊಸ ವರ್ಷ|ನ್ಯೂ ಇಯರ್", "01/01"],
            "Pongal": ["ಪೊಂಗಲ್", "14/01"], 
            "Makar Sankranti":["ಮಕರ ಸಂಕ್ರಾಂತಿ|ಸಂಕ್ರಾಂತಿ|ಸಂಕ್ರಮಣ", "14/01"],
            "Republic Day":	["ಗಣರಾಜ್ಯೋತ್ಸವ|ರಿಪಬ್ಲಿಕ್ ಡೇ", "26/01"],
            "Maha Shivaratri/Shivaratri": ["ಮಹಾ ಶಿವರಾತ್ರಿ|ಶಿವರಾತ್ರಿ","01/03"],
            "Holi":	["ಹೋಳಿ", "08/03"],
            "Ugadi": ["ಯುಗಾದಿ", "02/04"],
            "Ramzan Id/Eid-ul-Fitar": ["ರಂಜಾನ್","03/05"], 
            "Bakr Id/Eid ul-Adha": ["ಬಕ್ರೀದ್|ಈದ್","10/07"], 
            "Muharram/Ashura": ["ಮೊಹರಂ", "09/08"],
            "Raksha Bandhan": ["ರಕ್ಷಾ ಬಂಧನ|ರಕ್ಷಾಬಂಧನ|ರಾಖಿ ಹಬ್ಬ","11/08"],
            "Varamahalakshmi": ["ವರಮಹಾಲಕ್ಷ್ಮಿ","12/08"],
            "Independence Day":	["ಸ್ವಾತಂತ್ರ್ಯ ದಿನಾಚರಣೆ|ಸ್ವಾತಂತ್ರ್ಯೋತ್ಸವ|ಇಂಡಿಪೆಂಡೆನ್ಸ್ ಡೇ","15/08"],
            "Janmashtami":	["ಕೃಷ್ಣ ಅಷ್ಟಮಿ|ಕೃಷ್ಣಾಷ್ಟಮಿ|ಜನ್ಮಾಷ್ಟಮಿ","19/08"],
            "Ganesh Chaturthi":	["ವಿನಾಯಕ ಚೌತಿ|ವಿನಾಯಕ ಚತುರ್ಥಿ|ಗಣೇಶ ಚತುರ್ಥಿ|ಗಣೇಶ ಚೌತಿ|ಚೌತಿ ಹಬ್ಬ","31/09"],
            "Onam":	["ಓಣಂ","08/09"],
            "First Day of Sharad Navratri":	["ನವರಾತ್ರಿ|ದುರ್ಗಾ ಪೂಜಾ","26/09"],	
            "Mahatma Gandhi Jayanti": ["ಗಾಂಧಿ ಜಯಂತಿ", "02/09"],
            "Dussehra,Vijaya dashami,Dasara": ["ದಸರಾ|ವಿಜಯ ದಶಮಿ|ವಿಜಯದಶಮಿ", "05/10"],
            "Diwali/Deepavali":	["ದೀಪಾವಳಿ|ದಿವಾಳಿ", "24/10"],
            "Christmas":	["ಕ್ರಿಸ್ಮಸ್", "25/12"],
            "Shri Rama Navami":	["ಶ್ರೀ ರಾಮ ನವಮಿ|ಶ್ರೀ ರಾಮನವಮಿ|ರಾಮನವಮಿ ", "30/03"]
        }

    if lang == "telugu":
        return {
            "New Year\'s Day": ["కొత్త సంవత్సరం|న్యూ ఇయర్", "31/01"],
            "Makar Sankranti": ["సంక్రాంతి", "14/01"], 
            "Republic Day": ["రిపబ్లిక్ డే|జెండా వందనం|జెండా పండుగ|జెండామందరం", "26/01"],
            "Maha Shivaratri/Shivaratri": ["మహా శివరాత్రి|శివరాత్రి", "01/03"], 
            "Holi": ["హోలీ|హ్ోలీ","08/03"],
            "Ugadi": ["ఉగాది","02/04"], 
            "Gudi Padwa": ["గుడి పడ్వా", "02/04"],
            "Ramzan Id/Eid-ul-Fitar (Tentative Date)": ["రంజాన్", "03/05"], 
            "Bakr Id/Eid ul-Adha (Tentative Date)": ["బక్రీద్","10/07"],
            "Muharram/Ashura (Tentative Date)": ["మొహర్రం|ముహూర్రం", "09/08"], 
            "Raksha Bandhan": ["రక్ష బంధన్|రక్షబంధన్|రాఖీ పౌర్ణమి", "11/08"],
            "Independence Day": ["ఇండిపెండెన్స్ డే|జెండా వందనం|జెండా పండుగ|జెండామందరం", "15/08"],
            "Janmashtami": ["కృష్ణ అష్టమి|కృష్ణాష్టమి|జన్మాష్టమి", "18/08"],
            "Ganesh Chaturthi": ["వినాయక చవితి|వినాయక చతుర్థి|గణేష్ చతుర్థి|గణేష్ చవితి", "31/08"],
            "Mahatma Gandhi Jayanti": ["గాంధీ జయంతి", "02/09"], 
            "Dussehra": ["దసరా|దసర|విజయ దశమి|విజయదశమి","05/10"], 
            "Diwali": ["దీపావళి|దివాలి|దివాళి", "24/10"],
            "Christmas": ["క్రిస్మస్", "25/12"],
            "Shri Rama Navami": ["శ్రీ రామ నవమి|శ్రీ రామనవమి|రామనవమి|సీత రాముల కళ్యాణం|సీతరాముల కళ్యాణం", "30/03"],
            "Holi Ekadashi": ["తొలి ఏకాదశి", "07/03"], 
        }

    if lang == "marathi":
        return {
            "New Year\'s Day": ["नवीन वर्ष", "31/01"],
            "Guru Govind Singh Jayanti": ["गुरु गोविंद सिंग जयंती", "09/01"], 
            "Makar Sankranti": ["मकर संक्रांत", "14/01"],
            "Republic Day": ["गणतंत्र दिवस", "26/01"],
            "Maha Shivaratri/Shivaratri": ["महा शिवरात्री", "01/03"], 
            "Holi": ["होळी","08/03"],
            "Gudi Padwa": ["गुढी पाडवा", "02/04"],
            "Good Friday": ["गुड फ्रायडे", "07/04"],
            "Ramzan Id/Eid-ul-Fitar (Tentative Date)": ["रमजान", "03/05"], 
            "Bakr Id/Eid ul-Adha (Tentative Date)": ["बकरी ईद","10/07"],
            "Muharram/Ashura (Tentative Date)": ["मुहर्रम", "09/08"], 
            "Raksha Bandhan": ["रक्षा बंधन", "11/08"],
            "Independence Day": ["स्वातंत्र दिवस", "15/08"],
            "Janmashtami": ["जन्माष्टमी", "19/08"],
            "Ganesh Chaturthi": ["गणेश चतुर्थी", "31/08"],
            "Mahatma Gandhi Jayanti": ["महात्मा गांधी जयंती", "02/09"], 
            "Ram Navami": ["राम नवमी", "30/03"],
            "Dussehra": ["विजया दशमी","04/10"], 
            "Diwali": ["दिवाळी", "24/10"],
            "Bhai Duj": ["भाऊबीज", "26/10"],
            "Christmas": ["क्रिसमस", "25/12"],
            "Guru Nanank Jayanti": ["गुरु नानक जयंती", "08/11"],
            "Guru Govind Singh Jayanti": ["गुरु गोविंद सिंग जयंती","29/12"],
            "Holi Ekadashi": ["राम नवमी", "07/03"],
            "HANUMAN JAYANTI": ["हनुमान जयंती", "06/04"], 
        }

    if lang == "malayalam":
        return {
            "New Year\'s Day": ["പുതുവർഷ ദിനം|ന്യൂ ഇയർ ഡേ", "01/01"],
            "Pongal": ["പൊങ്കൽ", "14/01"], 
            "Makar Sankranti":["മകര സംക്രാന്തി|മകരവിളക്ക്", "14/01"],
            "Republic Day":	["റിപ്പബ്ലിക് ദിനം", "26/01"],
            "Maha Shivaratri/Shivaratri": ["മഹാ ശിവരാത്രി|ശിവരാത്രി","01/03"],
            "Holi":	["ഹോളി", "08/03"],
            "Vaisakhi": ["വിഷു", "14/04"],
            "Good Friday": ["ദുഃഖവെള്ളി", "07/04"],
            "Easter Day": ["ഈസ്റ്റർ ദിനം|ഈസ്റ്റർ|ഈസ്റ്റർ ഞായർ", "17/01"],
            "Ramzan Id/Eid-ul-Fitar": ["റംസാൻ|ഈദുൽ ഫിത്തർ|ചെറിയ പെരുന്നാൾ","03/05"], 
            "Bakr Id/Eid ul-Adha": ["ബക്രീദ്|വലിയ പെരുന്നാൾ","10/07"], 
            "Muharram/Ashura": ["മുഹറം", "09/08"],
            "Independence Day":	["സ്വാതന്ത്യദിനം","15/08"],
            "Janmashtami":	["ശ്രീകൃഷ്ണ ജയന്തി|ജന്മാഷ്ടമി","19/08"],
            "Ganesh Chaturthi":	["വിനായക ചതുർത്ഥി|ഗണേശ ചതുർത്ഥി","31/08"],
            "Onam":	["ഓണം","08/09"],	
            "Mahatma Gandhi Jayanti": ["ഗാന്ധി ജയന്തിತಿ", "02/09"],
            "Maha Navami": ["മഹാ നവമി", "04/09"],
            "Dussehra,Vijaya dashami,Dasara": ["വിജയ ദശമി", "05/10"],
            "Diwali/Deepavali":	["ദീപാവലി", "24/10"],
            "Christmas":	["ക്രിസ്മസ്", "25/12"]
        }

    if lang == "bangla":
        return {
            "New Year\'s Day": ["নববর্ষ", "01/01"],
            "Makar Sankranti":["মকর সংক্রান্তি", "14/01"],
            "Republic Day":	["প্রজাতন্ত্র দিবস", "26/01"],
            "Maha Shivaratri/Shivaratri": ["শিব রাত্রি","01/03"],
            "Holi":	["দোল|হোলি", "08/03"],
            "Vaisakhi": ["পয়লা বৈশাখ", "14/04"],
            "Good Friday": ["গুড ফ্রাইডে", "07/04"],
            "Easter Day": ["ഈസ്റ്റർ ദിനം | ഈസ്റ്റർ | ഈസ്റ്റർ ഞായർ", "17/01"],
            "Ramzan Id/Eid-ul-Fitar": ["রোজার ঈদ|ঈদ উল ফিতার","03/05"], 
            "Bakr Id/Eid ul-Adha": ["বোকরা ঈদ|কুরবানির ঈদ|কোরবাণীর ঈদ|কোরবানির ঈদ|কোরবাণী|কোরবানি","10/07"], 
            "Muharram/Ashura": ["মহরম", "09/08"],
            "Rakhi": ["রাখি|রাখিবন্ধন", "11/08"],
            "Independence Day":	["স্বাধীনতা দিবস","15/08"],
            "Janmashtami":	["জন্মাষ্টমি|জন্মাষ্টমী","19/08"],
            "Ganesh Chaturthi":	["গণেশ চতুর্থী","31/08"],
            "First Day of Sharad Navratri": ["মহালয়া", "26/09"],
            "First Day of Durga Puja Festivities": ["পঞ্চমী", "01/10"],
            "Mahatma Gandhi Jayanti": ["ষষ্ঠী|গান্ধী জয়ন্তী", "02/09"],
            "Maha Saptami": ["সপ্তমী", "02/09"],
            "Maha Ashtami": ["অষ্টমী", "03/09"],
            "Maha Navami": ["নবমী", "04/09"],
            "Dussehra,Vijaya dashami,Dasara": ["দশমী|বিজয় দশমী|বিজয়া দশমী", "05/10"],
            "Diwali/Deepavali":	["কালীপূজা|দিবালি|কালীপূজো", "24/10"],
            "Bhai Duj": ["ভাই ফোটা", "26/10"],
            "Chhat Puja" : ["ছটপুজো|ছট পুজো|ছট", "30/10"],
            "Guru Nanak Jayanti": ["গুরু নানক জয়ন্তী", "29/12"],
            "Christmas":	["ক্রিসমাস|বড়দিন", "25/12"]
        }

    if lang == "tamil":
        return {
            "New Year\'s Day": ["நியூ இயர்|புத்தாண்டு|புத்தாண்டு தினம்", "01/01"],
            "Pongal": ["பொங்கல்|தை", "14/01"], 
            "Makar Sankranti":["மகர சங்கராந்தி", "14/01"],
            "Republic Day":	["குடியரசு தினம்", "26/01"],
            "Maha Shivaratri/Shivaratri": ["மகா சிவராத்திரி|சிவராத்திரி","01/03"],
            "Holi":	["ஹோலி", "08/03"],
            "Ugadi": ["உகாதி", "02/04"],
            "Good Friday": ["புனித வெள்ளி", "07/04"],
            "Easter Day": ["ஈஸ்டர் தினம்|ஈஸ்டர்|ஈஸ்டர் ஞாயிறு", "17/01"],
            "Ramzan Id/Eid-ul-Fitar": ["ரம்ஜான் ஐத்|ஈத்-உல்-பிதார்","03/05"], 
            "Bakr Id/Eid ul-Adha": ["பக்ர் ஐத்|ஈத் உல்-அதா","10/07"], 
            "Muharram/Ashura": ["முஹர்ரம்|ஆஷுரா", "09/08"],
            "Rakhi": ["ரக்ஷா பந்தன்", "11/08"],
            "Independence Day":	["சுதந்திர தினம்","15/08"],
            "Janmashtami":	["ஜென்மாஷ்டமி","19/08"],
            "Ganesh Chaturthi":	["விநாயக சதுர்த்தி","31/08"],
            "Onam": ["ஓணம்", "08/09"],
            "Mahatma Gandhi Jayanti": ["காந்தி ஜெயந்தி", "02/09"],
            "Dussehra,Vijaya dashami,Dasara": ["தசரா|விஜய தசமி|தசரா", "05/10"],
            "Diwali/Deepavali":	["தீபாவளி|தீபாவளி", "24/10"],
            "Guru Nanak Jayanti": ["குருநானக் ஜெயந்தி", "29/12"],
            "Christmas":	["கிறிஸ்துமஸ்ন", "25/12"],
            "Hanuman Jayanti": ["அனுமன் ஜெயந்தி", "06/04"],
            "Saraswati Puja": ["சரஸ்வதி பூஜை", "05/02"]
        }

    if lang == "punjabi":
        return {
            "New Year's Day": ["નવા વર્ષના દિવસ| નવા વર્ષ|નુતન વરસ|નુતન વર્ષ", "01/01"],
            "Guru Govind Singh Jayanti": ["ગુરુ ગોવિંદ સિંહ જયંતી", "20/01"],
            "Lohri": ["લોરી", "13/01"],
            "Pongal/Makar Sankranti": ["પોંગલ|સંક્રાત|ઉતરાયણ|મકર સંક્રાંતિ|ખીહર|સંક્રાંત|સંકરાત", "14/01"],
            "Republic Day":	["રિપબ્લિક ડે|ગણતંત્ર દિવસ", "26/01"],
            "Lunar New Year": ["લુનાર નવું વર્ષ", "01/02"],
            "Vasant Panchami": ["વસંત પંચમી|ઋષિ પંચમી", "05/02"],
            "Valentine's Day":	["વેલેન્ટાઇન ડે", "14/01"],
            "Hazarat Ali's Birthday": ["હઝરત અલી જયંતી|હઝરત અલી ઉર્સ", "15/02"],
            "Guru Ravidas Jayanti": ["ગુરુ રવિદાસ જયંતી", "16/02"],
            "Shivaji Jayanti":	["શિવાજી જયંતી", "19/02"],
            "Maharishi Dayanand Saraswati Jayanti": ["મહર્ષિ દયાનંદ સરસ્વતી જયંતી", "26/02"],
            "Maha Shivaratri/Shivaratri": ["મહા શિવરાત્રી|શિવરાત્રી", "01/03"],
            "Holika Dahana": ["હોલિકા દહન", "17/03"],
            "Holi/Dolyatra": ["હોળી|હોલી|ડોલયાત્રા", "18/03"],
            "March Equinox": ["માર્ચ એક્વિનોક્ષ", "20/03"],
            "Chaitra Sukhladi/Ugadi/Gudi Padwa":	["ચૈત્રી નવરાત્રી|ઉગાડી|ગુડી પડવો", "02/04"],
            "Rama Navami": ["રામ નવમી|રામ નોમ", "10/04"],
            "Maundy Thursday/Mahavir Jayanti/Vaisakhi/Mesadi / Vaisakhadi/Ambedkar Jayanti":	["મૌનડી થર્સડે|મહાવીર જયંતી|વૈશાખી|મેસાડી|વૈસાખડી|બાબા સાહેબ જયંતી| આંબેડકર જયંતી|બાબા સાહેબ આંબેડકર જયંતી", "14/04"],
            "Good Friday":	["ગૂડ ફ્રાઈડે| ગૂડ ફ્રાયડે", "15/04"],
            "Easter Day": ["ઈસ્ટર", "17/04"],
            "Jamat Ul-Vida (Tentative Date)": ["જમાત ઉલ વિદા", "29/04"],
            "Ramzan Id/Eid-ul-Fitar (Tentative Date)": ["રમજાન ઈદ|રમઝાન ઈદ|ઈદ ઉલ ફિતર", "03/05"],
            "Mother's Day": ["મધર્સ ડે", "08/05"],
            "Birthday of Ravindranath":	["રવીન્દ્રનાથ જયંતી|ટાગોર જયંતી", "09/05"],
            "Buddha Purnima/Vesak":	["બુદ્ધ પૂર્ણિમા|વેસક", "16/05"],
            "Father's Day":	["ફાધર્સ ડે", "19/06"],
            "Rath Yatra": ["રથ યાત્રા|અષાઢી બીજ", "01/07"],
            "Bakr Id/Eid ul-Adha (Tentative Date)":	["બકર ઈદ|બકરી ઈદ|ઈદ ઉલ અધા", "10/07"],
            "Guru Purnima":	["ગુરુ પૂર્ણિમા|ગુરુ પૂનમ", "13/07"],
            "Friendship Day": ["ફ્રેન્ડશીપ ડે", "07/08"],
            "Muharram/Ashura (Tentative Date)":	["મોહરમ|મોરમ|અશુરા", "09/08"],
            "Raksha Bandhan (Rakhi)": ["રક્ષાબંધન|બળેવ", "11/08"],
            "Independence Day":	["સ્વતંત્ર દિન|સ્વતંત્ર દિવસ|ઈન્ડીપેન્ડંસ ડે", "15/08"],
            "Parsi New Year": ["પારસી ન્યુ યર| પારસી નવા વર્ષ", "16/08"],
            "Janmashtami (Smarta)":	["જન્માષ્ટમી|સાતમ આઠમ", "18/08"],
            "Janmashtami": ["જન્માષ્ટમી", "19/08"],
            "Ganesh Chaturthi/Vinayaka Chaturthi":	["ગણેશ ચતુર્થી|ગણેશ ચોથ|વિનાયક ચતુર્થી", "31/08"],
            "Onam": ["ઓનમ", "08/09"],
            "September Equinox":	["સપ્ટેમ્બર એક્વિનોક્ષ", "23/09"],
            "First Day of Sharad Navratri":	["શરદ નવરાત્રી", "26/09"],
            "First Day of Durga Puja Festivities": ["દુર્ગાપૂજા", "01/10"],
            "Mahatma Gandhi Jayanti": ["મહાત્મા ગાંધી જયંતી|ગાંધી જયંતી", "02/10"],
            "Maha Saptami": ["મહા સપ્તમી", "02/10"],
            "Maha Ashtami":	["મહા અષ્ટમી", "03/10"],
            "Maha Navami":	["મહા નવમી", "04/10"],
            "Dussehra":	["દશેરા", "05/10"],
            "Maharishi Valmiki Jayanti/Milad un-Nabi/Id-e-Milad": ["મહર્ષિ વાલ્મીકી જયંતી|મિલાદ ઉલ નબી|ઈદે મિલાદ|ઈદ", "09/10"],
            "Karaka Chaturthi (Karva Chauth)":	["કરવા ચોથ|કડવા ચોથ", "13/10"],
            "Naraka Chaturdasi/Diwali / Deepavali": ["નર્ક ચતુર્દશી|કાળી ચૌદસ|દિવાળી|દીપાવલી", "24/10"],
            "Govardhan Puja":	["ગોવર્ધન પૂજા", "25/10"],
            "Bhai Duj":	["ભાઈ બીજ", "26/10"],
            "Chhat Puja (Pratihar Sashthi/Surya Sashthi)":	["છઠ પૂજા", "30/10"],
            "Halloween": ["હેલોવીન", "31/10"],
            "Guru Nanak Jayanti": ["ગુરુ નાનક જયંતી", "08/11"],
            "Guru Tegh Bahadur's Martyrdom Day": ["ગુરુ તેગ બહાદુર શહીદ દિવસ", "24/11"],
            "First Day of Hanukkah": ["હનાક્કા", "19/12"],
            "Christmas Eve": ["ક્રિસમસ", "25/12"],
            "New Year's Eve": ["નવા વરસ|નવું વરસ|નવા વર્ષ|નવ વર્ષ", "31/12"],
            "Dhuleti": ["ધૂળેટી|પડવો", "08/03"],
            "sharad punam":	["શરદ પૂનમ", "09/10"],   
        }
    
    if lang == "gujarati":
        return {
            "New Year's Day": ["નવા વર્ષના દિવસ| નવા વર્ષ|નુતન વરસ|નુતન વર્ષ", "01/01"],
            "Guru Govind Singh Jayanti": ["ગુરુ ગોવિંદ સિંહ જયંતી", "20/01"],
            "Lohri": ["લોરી", "13/01"],
            "Pongal/Makar Sankranti": ["પોંગલ|સંક્રાત|ઉતરાયણ|મકર સંક્રાંતિ|ખીહર|સંક્રાંત|સંકરાત", "14/01"],
            "Republic Day":	["રિપબ્લિક ડે|ગણતંત્ર દિવસ", "26/01"],
            "Lunar New Year": ["લુનાર નવું વર્ષ", "01/02"],
            "Vasant Panchami": ["વસંત પંચમી|ઋષિ પંચમી", "05/02"],
            "Valentine's Day":	["વેલેન્ટાઇન ડે", "14/01"],
            "Hazarat Ali's Birthday": ["હઝરત અલી જયંતી|હઝરત અલી ઉર્સ", "15/02"],
            "Guru Ravidas Jayanti": ["ગુરુ રવિદાસ જયંતી", "16/02"],
            "Shivaji Jayanti":	["શિવાજી જયંતી", "19/02"],
            "Maharishi Dayanand Saraswati Jayanti": ["મહર્ષિ દયાનંદ સરસ્વતી જયંતી", "26/02"],
            "Maha Shivaratri/Shivaratri": ["મહા શિવરાત્રી|શિવરાત્રી", "01/03"],
            "Holika Dahana": ["હોલિકા દહન", "17/03"],
            "Holi/Dolyatra": ["હોળી|હોલી|ડોલયાત્રા", "18/03"],
            "March Equinox": ["માર્ચ એક્વિનોક્ષ", "20/03"],
            "Chaitra Sukhladi/Ugadi/Gudi Padwa":	["ચૈત્રી નવરાત્રી|ઉગાડી|ગુડી પડવો", "02/04"],
            "Rama Navami": ["રામ નવમી|રામ નોમ", "10/04"],
            "Maundy Thursday/Mahavir Jayanti/Vaisakhi/Mesadi / Vaisakhadi/Ambedkar Jayanti":	["મૌનડી થર્સડે|મહાવીર જયંતી|વૈશાખી|મેસાડી|વૈસાખડી|બાબા સાહેબ જયંતી| આંબેડકર જયંતી|બાબા સાહેબ આંબેડકર જયંતી", "14/04"],
            "Good Friday":	["ગૂડ ફ્રાઈડે| ગૂડ ફ્રાયડે", "15/04"],
            "Easter Day": ["ઈસ્ટર", "17/04"],
            "Jamat Ul-Vida (Tentative Date)": ["જમાત ઉલ વિદા", "29/04"],
            "Ramzan Id/Eid-ul-Fitar (Tentative Date)": ["રમજાન ઈદ|રમઝાન ઈદ|ઈદ ઉલ ફિતર", "03/05"],
            "Mother's Day": ["મધર્સ ડે", "08/05"],
            "Birthday of Ravindranath":	["રવીન્દ્રનાથ જયંતી|ટાગોર જયંતી", "09/05"],
            "Buddha Purnima/Vesak":	["બુદ્ધ પૂર્ણિમા|વેસક", "16/05"],
            "Father's Day":	["ફાધર્સ ડે", "19/06"],
            "Rath Yatra": ["રથ યાત્રા|અષાઢી બીજ", "01/07"],
            "Bakr Id/Eid ul-Adha (Tentative Date)":	["બકર ઈદ|બકરી ઈદ|ઈદ ઉલ અધા", "10/07"],
            "Guru Purnima":	["ગુરુ પૂર્ણિમા|ગુરુ પૂનમ", "13/07"],
            "Friendship Day": ["ફ્રેન્ડશીપ ડે", "07/08"],
            "Muharram/Ashura (Tentative Date)":	["મોહરમ|મોરમ|અશુરા", "09/08"],
            "Raksha Bandhan (Rakhi)": ["રક્ષાબંધન|બળેવ", "11/08"],
            "Independence Day":	["સ્વતંત્ર દિન|સ્વતંત્ર દિવસ|ઈન્ડીપેન્ડંસ ડે", "15/08"],
            "Parsi New Year": ["પારસી ન્યુ યર| પારસી નવા વર્ષ", "16/08"],
            "Janmashtami (Smarta)":	["જન્માષ્ટમી|સાતમ આઠમ", "18/08"],
            "Janmashtami": ["જન્માષ્ટમી", "19/08"],
            "Ganesh Chaturthi/Vinayaka Chaturthi":	["ગણેશ ચતુર્થી|ગણેશ ચોથ|વિનાયક ચતુર્થી", "31/08"],
            "Onam": ["ઓનમ", "08/09"],
            "September Equinox":	["સપ્ટેમ્બર એક્વિનોક્ષ", "23/09"],
            "First Day of Sharad Navratri":	["શરદ નવરાત્રી", "26/09"],
            "First Day of Durga Puja Festivities": ["દુર્ગાપૂજા", "01/10"],
            "Mahatma Gandhi Jayanti": ["મહાત્મા ગાંધી જયંતી|ગાંધી જયંતી", "02/10"],
            "Maha Saptami": ["મહા સપ્તમી", "02/10"],
            "Maha Ashtami":	["મહા અષ્ટમી", "03/10"],
            "Maha Navami":	["મહા નવમી", "04/10"],
            "Dussehra":	["દશેરા", "05/10"],
            "Maharishi Valmiki Jayanti/Milad un-Nabi/Id-e-Milad": ["મહર્ષિ વાલ્મીકી જયંતી|મિલાદ ઉલ નબી|ઈદે મિલાદ|ઈદ", "09/10"],
            "Karaka Chaturthi (Karva Chauth)":	["કરવા ચોથ|કડવા ચોથ", "13/10"],
            "Naraka Chaturdasi/Diwali / Deepavali": ["નર્ક ચતુર્દશી|કાળી ચૌદસ|દિવાળી|દીપાવલી", "24/10"],
            "Govardhan Puja":	["ગોવર્ધન પૂજા", "25/10"],
            "Bhai Duj":	["ભાઈ બીજ", "26/10"],
            "Chhat Puja (Pratihar Sashthi/Surya Sashthi)":	["છઠ પૂજા", "30/10"],
            "Halloween": ["હેલોવીન", "31/10"],
            "Guru Nanak Jayanti": ["ગુરુ નાનક જયંતી", "08/11"],
            "Guru Tegh Bahadur's Martyrdom Day": ["ગુરુ તેગ બહાદુર શહીદ દિવસ", "24/11"],
            "First Day of Hanukkah": ["હનાક્કા", "19/12"],
            "Christmas Eve": ["ક્રિસમસ", "25/12"],
            "New Year's Eve": ["નવા વરસ|નવું વરસ|નવા વર્ષ|નવ વર્ષ", "31/12"],
            "Dhuleti": ["ધૂળેટી|પડવો", "08/03"],
            "sharad punam":	["શરદ પૂનમ", "09/10"],
        }

class EventExtract:

    def update_current_date(self, current_date, new_date, year):
        new_date = Date.strptime(new_date, '%d/%m/%Y')
        new_date = new_date.replace(tzinfo = pytz.timezone('Asia/Kolkata'))
        if current_date > new_date:
            year += 1
        new_date = new_date.strftime("%d/%m/%Y")
        return new_date + "/" + str(year)

class EventExtraction:
    
    def update_current_date(self, current_date, new_date, year):
        new_date = Date.strptime(new_date, '%d/%m/%Y')
        new_date = new_date.replace(tzinfo = pytz.timezone('Asia/Kolkata'))
        if current_date > new_date:
            year += 1
        new_date = new_date.strftime("%d/%m/%Y")
        return new_date 
    
    def extract_festivals(self, text, current_date, lang = "hindi"):
        try:
            # text = " ".join(text[0])
            year = current_date.year
            event_dict = event_list(lang)
            for events in event_dict:
                event_name = event_dict[events][0]
                dates = event_dict[events][1]
                event = re.search(event_name, text)
                if event != None and event.group() != "":
                    new_date = dates + "/"+ str(year)
                    return self.update_current_date(current_date=current_date, new_date=new_date, year=year)
            return -1
        except Exception as ex:
            return -1