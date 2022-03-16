# Selenium
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# Time
import time
# Json
import json

driver = Chrome('C:\\Selenium\\chromedriver.exe')
driver.get("https://country-code.cl/")

countries_to_continents_location = "data/countries_to_continents.json"

countries_to_continents = {}
for i in range(249):
    tds = driver.find_element(By.ID, f"row{i}").find_elements(By.TAG_NAME, "td")
    print(tds[0].text, tds[3].text)
    countries_to_continents[tds[3].text.strip()] = tds[0].text.strip()

# Save Json
json.dump(countries_to_continents, open(countries_to_continents_location, "w"), indent = 6)
