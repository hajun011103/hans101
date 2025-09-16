import os
import time
import urllib
import requests
from tqdm import tqdm

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

import config

def GoogleImageDownload(query, image_count, download_path):
    # Web driver
    service = Service('/usr/local/bin/chromedriver')
    driver = webdriver.Chrome(service=service)

    # Open the Browser
    URL = f"https://www.google.com/search?udm=2&q={query}"
    driver.get(URL)

    # Scroll to load more images
    for i in range(5):  # Scroll 3 times
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # Get Page Source
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    print(f"=== Start Collecting Images for {query} ===")
    
    # Get Image Elements
    images = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")
    img_urls = []
    for image in images:
        if len(img_urls) >= image_count:
            break
        src = image.get_attribute('src')
        if src and src.startswith('http') and not src.startswith('https://www.google.com/') and not src.startswith('data:image/'):
            img_urls.append(src)

    download_count = 0

    for img_url in tqdm(img_urls, desc=f"Downloading {query}"):
        
        # File Path
        file_path = os.path.join(download_path)
        
        # Download & Save Image
        urllib.request.urlretrieve(img_url, file_path + f"{download_count}.jpg")
        download_count += 1
        time.sleep(1)

    print(f"=== Finish Downloading {download_count} Images for {query} ===")
    driver.quit()

if __name__ == "__main__":
    cat_list = ["SiameseCat", "TuxedoCat", "NorwegianForestCat", "RussianBlueCat"]
    image_count = 100

    for cat in cat_list:
        download_path = os.path.join(config.TRAIN_DIR + '/' + cat + "/")
        GoogleImageDownload(cat, image_count, download_path)