import pandas as pd
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

from utils.constants import LAST_PLAYED_GAMEWEEK


def scrape_key_passes_data(gw_start=1, gw_end=LAST_PLAYED_GAMEWEEK) -> None:
    service = Service()
    options = ChromeOptions()
    options.add_argument("--headless")

    browser = webdriver.Chrome(service=service, options=options)
    browser.implicitly_wait(20)

    for gw_num in range(gw_start, gw_end + 1):
        gw_df = None
