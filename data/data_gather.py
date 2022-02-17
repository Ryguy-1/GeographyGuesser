# Selenium
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# Time
import time
# Wget
import wget
# Check if land or sea
from mpl_toolkits.basemap import Basemap
bm = Basemap()

# Latitude -> -90 to 90
# Longitude -> -180 to 180
start_lat = -90
end_lat = 90
start_long = -180
end_long = 180
increment = 0.3

# Loading Wait Time
load_time = 3

# Driver Info
window_width = 1200
window_height = 800

def initialize_scraper(chromedriver_path):
    # Initialize Driver Instance
    driver = Chrome(chromedriver_path)
    # Get Google Maps
    driver.get("https://www.google.com/maps/@23.8869203,5.4159443,3z")
    # Maximize Window
    driver.set_window_size(window_width, window_height)
    # Begin Sending Keys
    
    # Iterate over Longitude
    for i in range(start_lat*10, end_lat*10, int(increment*10)):
        for j in range(start_long*10, end_long*10, int(increment*10)):
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
            
    # Wait for the page to load
    time.sleep(load_time)

    
def download_image(lat, long, driver):
    # Download Image to Image Folder
    try:
        wget.download(driver.find_element(By.XPATH, "//*[@id=\"pane\"]/div/div[1]/div/div/div[1]/div[1]/button/img").get_attribute("src"), "data/images/" + str(lat) + "-" + str(long) + ".png")
    except:
        print("No Image Found")



if __name__ == "__main__":
    initialize_scraper("C:/Selenium/chromedriver.exe")