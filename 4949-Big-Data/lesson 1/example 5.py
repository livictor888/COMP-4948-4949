from selenium.webdriver.chrome.service import Service
import re
import time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By


DRIVER_PATH    = "D:\Downloads\chromedriver_win32\chromedriver.exe"              # Windows
browser = None
URL = "https://vpl.bibliocommons.com/events/search/index"

browser = webdriver.Chrome(service=Service(DRIVER_PATH))
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)
libraryEvents = browser.find_elements(By.CSS_SELECTOR,
                            ".events-details-container")

# Uses BeautifulSoup and Regex to remove HTML and
# unwanted characters.
def removeHtmlAndUnwantedCharacters(contents):
    htmlRemoved = contents.get_attribute('innerHTML')

    # Beautiful soup allows us to remove HTML tags from our content if it exists.
    soup = BeautifulSoup(htmlRemoved, features="lxml")
    textOnly = soup.get_text()

    # Remove hidden carriage returns and tabs.
    textOnly = re.sub(r"[\n\t]*", "", textOnly)

    # Replace two or more consecutive empty spaces with '*'
    textOnly = re.sub('[ ]{2,}', '*', textOnly)
    return textOnly

def extractTime(startTimePosition, content):
    endPosition = -1
    firstHypenPosition = content.find(" – ")
    for i in range(firstHypenPosition, len(content)):
        substring = content[i:i + 2]
        if (substring == "AM" or substring == "PM"):
            endPosition = i + 2
            break
    timeString = content[startTimePosition:endPosition]
    return timeString

def extractTitle(content):
    # Title occurs right after 1st number and ends at 2nd number.
    # Example: "Summer Reading Club Celebration" is found in
    # "Aug31Summer Reading Club Celebration3:00 P"

    startNumberFound = False
    startPosition = -1
    endTitlePosition = -1

    for i in range(0, len(content)):
        character = content[i]
        try:
            parsedNumber = int(character)
            if (not startNumberFound):
                startNumberFound = True
            elif startPosition != -1:
                endTitlePosition = i
                break
        except:
            if (startNumberFound and startPosition == -1):
                startPosition = i

    title = content[startPosition:endTitlePosition]
    return title, endTitlePosition

def getDate(content):
    barPosition = content.find("|")
    onPosition = -1
    for i in range(barPosition, len(content)):
        substring = content[i:i + 2]
        if (substring == 'on'):
            onPosition = i
            break
    dateString = content[barPosition + 2:onPosition]
    return dateString

for libraryEvent in libraryEvents:
    textContent = removeHtmlAndUnwantedCharacters(libraryEvent)
    print(textContent)
    title, endTitlePosition = extractTitle(textContent)
    timeString = extractTime(endTitlePosition, textContent)
    dateString = getDate(textContent)
    print("Title: " + title)
    print("Time: " + timeString)
    print("Date: " + dateString)
    print("***")  # Go to new line.
