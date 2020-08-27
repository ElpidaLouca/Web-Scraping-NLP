## Natural Language Processing Project

# useful packages
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup


class Web_scrapping(object):

    def web_scrapping(self):
        # access the initial website

        url = 'https://en.wikipedia.org/wiki/List_of_dinosaur_genera'

        # get the names of the links you want to get information from

        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        din_web_names = {}
        for link in soup.find_all("a"):
            url = link.get("href", "")
            if "/wiki/" in url:
                din_web_names[link.text.strip()] = url

        dinosaur_names = list(din_web_names.values())

        # eliminate website selection

        check = "/"
        dinosaur_names = [idx for idx in dinosaur_names if idx[0].lower() == check.lower()]
        names = []
        for i in range(len(dinosaur_names)):
            names.append(dinosaur_names[i].partition("/wiki/")[2])

        dinosaur_choice = names

        ########### EXTRACT MAIN BODY ##############

        # get access to all documents and extract the main body by following the path from inspect command

        documents_texts = []
        url = 'https://en.wikipedia.org/wiki/'
        chrome_options = webdriver.Chrome(ChromeDriverManager().install())

        ## collect all the information from the documents automatically

        for i in range(len(dinosaur_choice)):
            chrome_options.get(url + dinosaur_choice[i])
            main = ""
            for h in range(100):
                try:
                    main = main + (chrome_options.find_element_by_xpath(
                        'html/body/div[3]/div[3]/div[4]/div/' + "p[" + str(h) + "]").text)
                except:
                    pass
            documents_texts.append(main)

        ## CREATE EXCEL FILE OF MAIN BODY ##
        data = pd.DataFrame(list(zip(dinosaur_choice, documents_texts)), columns=["Dinosaur Names", "Documents"])
        data.to_csv("Wikipedia Text.csv")
        return data


x = Web_scrapping()
x.web_scrapping()
