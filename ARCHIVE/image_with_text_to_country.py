from google.cloud import vision
import io
import os
from spacy_langdetect import LanguageDetector
import spacy
from spacy.language import Language
import json
# Text to Location
from geopy.geocoders import Photon
geolocator = Photon(user_agent="measurements")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Ryland Birchmeier\\Documents\\google_cloud_auth.json"

# Load Once
nlp = spacy.load("en_core_web_sm")
lang_detect = LanguageDetector()

# Countries in Geoguesser
geoguesser_countries = json.load(open('data/countries_in_geoguesser.json')).keys()

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

    print(texts)

    # Return 0 if no text
    if len(texts) == 0:
        return None
    return format_text(texts[0].description)


def get_lang_detector(nlp, name):
    return lang_detect

def get_language_from_text(text):
    doc = nlp(text)
    return doc._.language


# Text to Location Geocoding
locator = Photon(user_agent="myGeocoder")
def text_to_location(text):
    location = locator.geocode(query=text, exactly_one=False, limit=5)
    return location


# Define Once
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)


# Main Method for Image to Possible Country Locations
def text_country_identification(image_loc):
    text = detect_text(image_loc)
    print(text)
    if text is not None:
        # Location Object
        location = text_to_location(text)
        print(location)
        # Print Possible Locations
        print(f"Possible Location Found: {location}")
        # Convert Country Name to Language
        language_dict = get_language_from_text(location.address.split(",")[-1])
        # Get Language
        language_code = language_dict['language']
        # Print out Language Found
        print(f"Language Found: {language_code}")
        # Get Score
        score = language_dict['score']
        # Find Matching Countries to Language
        matched_countries = []
        # Get Mappings from Country Codes to Language Names
        language_mapping_dict = json.load(open('data/country_to_languages.json'))
        # Iterate
        for country_code, language_code_arr in language_mapping_dict.items():
            # For Languages
            for language in language_code_arr:
                # If Language Matches Text
                if language == language_code:
                    # If Country is in Geoguesser
                    if country_code in geoguesser_countries:
                        # Append Country to Possible Countries
                        matched_countries.append(country_code)
        return matched_countries, score, location
    else:
        return None, None, None


print(text_to_location("Dijon-Ville"))