import requests
from bs4 import BeautifulSoup

class Data:
    @staticmethod
    def  web_scraping(url):
        respone = requests.get(url).content
        soup = BeautifulSoup(respone, "html.parser")
        return soup.get_text()