import time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

DRIVER_PATH    = "D:\Downloads\chromedriver_win32\chromedriver.exe"              # Windows
browser = None
URL = "https://www.rottentomatoes.com/browse/movies_in_theaters/sort:newest?page=1"

browser = webdriver.Chrome(service=Service(DRIVER_PATH))
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

titles = browser.find_elements(By.CSS_SELECTOR, ".p--small")
dates = browser.find_elements(By.CSS_SELECTOR, ".smaller")

for t in titles:
    start = e.get_attribute('innerHTML')
    # Beautiful soup allows us to remove HTML tags from our content if it exists.
    soup = BeautifulSoup(start, features="lxml")
    print(soup.get_text())
    print("***") # Go to new line.

for d in dates:
    start = d.get_attribute('innerHTML')
    # Beautiful soup allows us to remove HTML tags from our content if it exists.
    soup = BeautifulSoup(start, features="lxml")
    print(soup.get_text())
    print("***") # Go to new line.
