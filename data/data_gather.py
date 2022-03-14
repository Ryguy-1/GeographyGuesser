# Selenium
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# Time
import time
# Wget
import wget
# Json
import json
# Check if land or sea
from mpl_toolkits.basemap import Basemap
bm = Basemap()
    
def download_image(lat, long, driver):
    # Download Image to Image Folder
    try:
        src = driver.find_element(By.XPATH, "//*[@id=\"pane\"]/div/div[1]/div/div/div[1]/div[1]/button/img").get_attribute("src")
        # Check if URL exists and is not in already downloaded list
        if src != "" and src not in urls_downloaded.keys():
            wget.download(src, "data/images/" + str(lat) + "_" + str(long) + ".png")
            urls_downloaded[src] = [lat, long]
            update_urls_json()
        else:
            print("Image Doesn't Exist or Already Downloaded")
    except:
        print("No Image Found")


def update_urls_json():
    with open(json_location, "w") as f:
        json.dump(urls_downloaded, f)

def load_urls_json():
    with open(json_location, "r") as f:
        return json.load(f)


def initialize_scraper(chromedriver_path):
    # Initialize Driver Instance
    driver = Chrome(chromedriver_path)
    # Get Google Maps
    driver.get("https://www.google.com/maps/@23.8869203,5.4159443,3z")
    # Maximize Window
    driver.set_window_size(window_width, window_height)
    # Begin Sending Keys
    
    # Iterate over Longitude
    for i in range(int(start_lat*10), int(end_lat*10), int(increment*10)):
        for j in range(int(start_long*10), int(end_long*10), int(increment*10)):
            # Latitude / Longitude
            lat = i/10; long = j/10
            if not bm.is_land(long, lat):
                continue
            # Get Search Bar
            search_bar_element = driver.find_element(By.ID, "searchboxinput")
            # Clear Search Bar
            search_bar_element.clear()
            # Click on Search Bar
            search_bar_element.click()
            # Send Long/Lat
            search_bar_element.send_keys(str(lat) + ", " + str(long))
            # Search
            search_bar_element.send_keys(Keys.ENTER)
            # Wait for Load
            time.sleep(load_time)
            # Download Thumbnail (600 x 400)
            download_image(lat, long, driver)
        # Reload Driver
        driver.quit()
        driver = Chrome(chromedriver_path)
        driver.get("https://www.google.com/maps/@23.8869203,5.4159443,3z")
        driver.set_window_size(window_width, window_height)
        time.sleep(load_time)

            
    # Wait for the page to load
    time.sleep(load_time)


# Latitude -> -90 to 90
# Longitude -> -180 to 180
start_lat = -90
end_lat = 90
start_long = -180
end_long = 180
increment = 0.3

# Loading Wait Time
load_time = 2.5

# Driver Info
window_width = 1200
window_height = 800

# Urls Downloaded Keep Track
json_location = "data/urls_downloaded.json"
# Load urls downloaded
urls_downloaded = load_urls_json()
print(f"Urls Downloaded: {len(urls_downloaded)}")


if __name__ == "__main__":
    initialize_scraper("C:/Selenium/chromedriver.exe")