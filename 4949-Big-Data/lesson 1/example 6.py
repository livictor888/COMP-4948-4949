# DRIVER_PATH = "C:\\datasets\\chromedriver.exe"
DRIVER_PATH    = "D:\Downloads\chromedriver_win32\chromedriver.exe"              # Windows

from selenium.webdriver.chrome.service import Service
import time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By

URL     = "https://animalfactguide.com/links/"
browser = webdriver.Chrome(service=Service(DRIVER_PATH))
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

libraryEvents = browser.find_elements(By.CSS_SELECTOR,
                        ".events-details-container")

# Find the search input.
search = browser.find_element(By.CSS_SELECTOR,"#s")
search.send_keys("Zebra")

# Find the search button - this is only enabled when a search query is entered
button = browser.find_element(By.CSS_SELECTOR, "#searchsubmit")
button.click()  # Click the button.

time.sleep(3)


# Extracts text from scraped content.
def extractText(data):
    text    = data.get_attribute('innerHTML')
    soup    = BeautifulSoup(text, features="lxml")
    content = soup.get_text()
    return content

titles       = browser.find_elements(By.CSS_SELECTOR,".entry-title a")
descriptions = browser.find_elements(By.CSS_SELECTOR, ".entry-summary p")

titleList       = []
descriptionList = []

for i in range(0, len(titles)):
    # extract title and add to list.
    title       = extractText(titles[i])
    titleList.append(title)

    # extract description and add to list.
    description = extractText(descriptions[i])
    descriptionList.append(description)

# Show the content.
for i in range(0, len(descriptionList)):
    print("\n********************")
    print("Title:       " + titleList[i])
    print("Description: " + descriptionList[i])

