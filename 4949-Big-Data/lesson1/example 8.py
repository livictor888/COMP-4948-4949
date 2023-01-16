# DRIVER_PATH = "C:\\datasets\\chromedriver.exe"
DRIVER_PATH    = "D:\Downloads\chromedriver_win32\chromedriver.exe"

import re
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

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



# Give the browser time to load all content.
time.sleep(3)
searchItems =\
    browser.find_elements(By.CSS_SELECTOR, ".bcit-search-results__title")

# Uses BeautifulSoup and Regex to remove HTML and
# unwanted characters.
def removeHtmlAndUnwantedCharacters(contents):
    htmlRemoved = contents.get_attribute('innerHTML')

    # Beautiful soup allows us to remove HTML tags from our content if it exists.
    soup = BeautifulSoup(htmlRemoved, features="lxml")
    textOnly = soup.get_text()

    # Remove hidden carriage returns and tabs.
    textOnly = re.sub(r"[\n\t]*", "", textOnly)

    # Replace two or more consecutive single space.
    textOnly = re.sub('[ ]{2,}', ' ', textOnly)
    return textOnly.strip() # remove leading and trailing spaces.

for searchItem in searchItems:
    textContent = removeHtmlAndUnwantedCharacters(searchItem)
    print(textContent)
