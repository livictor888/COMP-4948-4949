# DRIVER_PATH = "C:\\datasets\\chromedriver.exe"
DRIVER_PATH    = "D:\Downloads\chromedriver_win32\chromedriver.exe"              # Windows

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By

browser = None
URL     = "https://www.bcit.ca/"
browser = webdriver.Chrome(service=Service(DRIVER_PATH))
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

# Find the search input.
search = browser.find_element(By.CSS_SELECTOR,"#site-header-search")
search.send_keys("Analytics")

# Even when there is a search button sometimes it is not clickable.
# However, sometimes you can send a return key and this will initiate
# the search.
search.send_keys(Keys.RETURN)

