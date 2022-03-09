from google.cloud import vision
import io
import os
from spacy_langdetect import LanguageDetector
import spacy
from spacy.language import Language
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\rylan\\Documents\\google_cloud_auth.json"

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

    # Return 0 if no text
    if len(texts) == 0:
        return None
    return format_text(texts[0].description)


def get_lang_detector(nlp, name):
    return lang_detect

def get_language_from_text(text):
    
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)
    doc = nlp(text)
    return doc._.language


def text_country_identification(image_loc):
    text = detect_text(image_loc)
    print(text)
    if text is not None:
        language_dict = get_language_from_text(text)
        language_code = language_dict['language']
        # Find Matching Countries to Language
        matched_countries = []
        language_mapping_dict = json.load(open('data/country_to_languages.json'))
        for country_code, language_code_arr in language_mapping_dict.items():
            for language in language_code_arr:
                if language == language_code:
                    if country_code in geoguesser_countries:
                        matched_countries.append(country_code)
        return matched_countries
    else:
        return None

print(text_country_identification("live_images/image.jpg"))