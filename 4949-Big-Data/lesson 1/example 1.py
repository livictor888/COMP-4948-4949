from selenium import webdriver
from selenium.webdriver.chrome.service import Service
PATH    = "D:\Downloads\chromedriver_win32\chromedriver.exe"              # Windows

# This loads webdriver from the local machine if it exists.
try:
    driver = webdriver.Chrome(service=Service(PATH))
    print("Success! The path to webdriver_manager was found.")

# If a webdriver not found error occurs it is then downloaded.
except:
    print("webdriver not found. Update 'PATH' with file path in the download.")
