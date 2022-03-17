# Selenium
from re import L
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# OS
import os
# Glob
import glob
# OpenCV
import cv2
# Time
import time
# Json
import json
# Wget
import wget
# Traceback
import traceback
# Numpy
import numpy as np

# Google
driver = Chrome(service=Service(ChromeDriverManager().install()))
driver.maximize_window()

# Get Countries in Geoguesser
countries_in_geoguesser = json.load(open("data/countries_in_geoguesser.json", "r"))
print(countries_in_geoguesser)

# Helper (Scroll to End)
def scroll_to_end(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    try:
        driver.find_element(By.CLASS_NAME, "YstHxe").click()
        print("Pushed See More")
    except Exception:
        pass
    time.sleep(8)


# Keep Track of URLS Downloaded
urls_downloaded = {}

# Set Up Country Folder
def initialize_country_folder(country_name):
    if not os.path.exists(f"data/images_country_flags_original/{country_name}"):
        os.makedirs(f"data/images_country_flags_original/{country_name}")
        return True
    return False
    
for country, full_name in countries_in_geoguesser.items():
    # Set Up Folder (returns true if new and false if already exists (pass because already done/being done))
    is_new = initialize_country_folder(country_name=country)
    if not is_new:
        continue
    # Get Google
    driver.get("https://www.google.com")
    # Change to Images Tab
    driver.find_elements(By.CLASS_NAME, "gb_d")[1].click()
    # Wait for Change
    time.sleep(1.5)
    # Get Search Bar
    search_bar = driver.find_element(By.NAME, "q")
    # Click Search Bar
    search_bar.click()
    # Send Keys
    search_bar.send_keys(full_name + " flag flying")
    # Get Images Tab
    search_bar.send_keys(Keys.ENTER)
    # Wait for Load
    time.sleep(1)
    # Counter for File Names
    counter = 0
    # Keep Track of If Reached Page End
    reached_page_end = False
    # Initialize Last Height for Checking if At Bottom
    last_height = driver.execute_script("return document.body.scrollHeight")
    # Search for All Images
    while not reached_page_end:
        images = driver.find_elements(By.TAG_NAME, "img")
        for img in images:
            # Make Sure Not Previously Downloaded
            if img.get_attribute("src") in urls_downloaded.keys():
                continue
            try:
                # Try Downloading Image
                wget.download(img.get_attribute("src"), f"data/images_country_flags_original/{country}/{country}_{counter}.png")
                # Check Image Properties
                image = np.array(cv2.imread(f"data/images_country_flags_original/{country}/{country}_{counter}.png"))
                h, w, c = image.shape
                if h < 150 or w < 150:
                    os.remove(f"data/images_country_flags_original/{country}/{country}_{counter}.png")
                    continue
                # Increase Counter and Add to List / Update
                counter+=1
                urls_downloaded[img.get_attribute("src")] = 0
            except:
                pass
        
        # Scroll to End
        scroll_to_end(driver)

        # Check if At Bottom of Page
        new_height = driver.execute_script("return document.body.scrollHeight")
        if last_height == new_height:
            reached_page_end = True
            print("Reached Hard Bottom")
        else:
            last_height = new_height
