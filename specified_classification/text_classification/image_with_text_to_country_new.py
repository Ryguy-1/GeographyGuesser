import locale
from google.cloud import vision
import io
import os
from spacy_langdetect import LanguageDetector
import spacy
from spacy.language import Language
import json
# Text to Location
from geopy.geocoders import Nominatim

# Load Google Cridentials For Vision API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# For Language Detection
nlp = spacy.load("en_core_web_sm")
lang_detect = LanguageDetector()

def get_lang_detector(nlp, name):
    return lang_detect

def get_language_from_text(text):
    doc = nlp(text)
    return doc._.language

Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

# Text to Location Geocoding
locator = Nominatim(user_agent="nomatim-geocoder-1")
def text_to_location(text):
    location = locator.geocode(query=text, exactly_one=True)
    return location

# Countries in Geoguesser
geoguesser_countries = json.load(open('data/countries_in_geoguesser.json')).keys()

# Google Text Detection
def detect_text(path):
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    def format_text(text):
        words = text.split("\n")
        words = [word.strip("") for word in words]
        words = [word for word in words if word != ""]
        text = ""
        for word in words:
            text += word + " "
        text.strip()
        return text

    # Return 0 if no text
    if len(texts) == 0:
        return None, None, None

    # Phrases Individual
    phrase_list = [text.description for text in texts]
    # Language
    language = texts[0].locale
    # Return
    return format_text(texts[0].description), phrase_list, language


# 1) Detect All Text in Image
# 2) Identify Language of Text (given by google)
# 3) Run Each Piece of Text Through Geolocation and Return Locations with Country that Speaks that Language
#       Spacy Used to Convert Country Name to Country Code (Weird Use, but Works very Well)


def geolocation_and_language_from_image_location(image_loc, min_valid_phrase_length = 6):
    # Detect All Text In Image
    text_all, phrase_list, language = detect_text(image_loc)
    # If None, return None
    if text_all is None and phrase_list is None and language is None:
        return None, None
    print(f"Found Text: {text_all}")
    print(f"Language is {language}")
    # Delete All Phrases less than Specified Length
    phrase_list = [phrase for phrase in phrase_list if len(phrase) >= min_valid_phrase_length]
    # Countries Found (Parallel Lists for Locations and Languages)
    locations = []; languages = []
    # Iterate Through Long Enough Phrases
    for phrase in phrase_list:
        # Get Location
        location = text_to_location(phrase)
        # Append Location
        locations.append(location)
        # Check if Address Exists
        if location is not None and location.address.split(",")[-1] is not None:
            languages.append(get_language_from_text(location.address.split(",")[-1])['language'])
        else:
            languages.append(None)
        
    # Find Addresses that Match Google (Most Trusted) Language, and add to a list of potential addresses
    addresses_matching_google_language = []
    # Iterate Through Langauges From Addresses
    for i in range(len(languages)):
        # If Language Matches Google Language
        if languages[i] == language:
            # Append Potential Location Within Country
            addresses_matching_google_language.append(locations[i])
            
    return addresses_matching_google_language, language

