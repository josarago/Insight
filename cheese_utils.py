import os
import pandas as pd
import requests
from bs4 import BeautifulSoup


split_columns = ['aroma','flavour','rind','texture','type','ingredients']

ingredients = ['pasteurized','unpasteurized','buffalo','camel','cow',
     'donkey','goat','mare','moose','buffalo','sheep','water buffalo',
     'reindeer','yak'
    ]

# define the class cheese
class RawCheese:
    """
        given a cheese name (as extracted from the list on cheese.com)
        one can access a dict of properties
    """
    def __init__(self,cheese_name):
        self.url_name = cheese_name
        self.cheese_com_url = 'https://www.cheese.com/{}/'.format(cheese_name)
        self.dump = []
        self.soup = None

    def get_soup(self):
        """
        get the BeautifulSoup object from the cheese url on cheese.com
        """
        if not self.soup:
            r = requests.get(self.cheese_com_url)
            self.soup = BeautifulSoup(r.content, 'html.parser')

    @property
    def dict(self):
        self.get_soup()
        cheese_dict = {'name': self.url_name}
        summary_points = self.soup.find_all("ul", {"class": "summary-points"})
        features_list = [x.text for x in summary_points[0].find_all("p")]
        for feature in features_list:
            if ":" in feature:
                split_feature = feature.split(":")
                feature_key = split_feature[0].lower().replace(" ","_")
                feature_val = split_feature[1].strip().lower()
                cheese_dict[feature_key]=feature_val
            elif "Made from" in feature:
                feature_key = 'ingredients'
                feature_val = feature.split("Made from")[1].lower()
                cheese_dict[feature_key]=feature_val.strip()
            else:
                self.dump.append(feature)
        return cheese_dict



def get_cheeses_url_names(soup):
    """
        given a soup, extract all the cheeses url names present in the corresponding page
    """
    cheeses_url_names = []
    cheeses_divs = soup.find_all("div", {"class": "col-sm-6 col-md-4 cheese-item text-center"})
    for cheese_div in cheeses_divs:
        cheeses_url_names.append(cheese_div.find("h3").findChild().attrs['href'].replace("/",""))
    return cheeses_url_names

def get_all_cheeses_url_names():
    """
        crawls cheese.com to extract all cheeses names
    """
    all_cheeses_url_names = set()
    # Create alphabet list of lowercase letters
    alphabet = []
    for letter in range(97,123):
        alphabet.append(chr(letter))
    # for each letter in the alphabet
    for letter in alphabet:
        keep_going=True
        n=0
        print(letter)
        while keep_going:
            n+=1
            letter_url = "https://www.cheese.com/alphabetical/?per_page=100&i={}&page={}#top".format(letter,n)
            clear_output(wait=True)
            display("getting letter {} page #{}".format(letter,n))
            time.sleep(0.1)
            r = requests.get(letter_url)
            soup = BeautifulSoup(r.content,'html.parser')
            page_cheeses_url_names = set(get_cheeses_url_names(soup))
            diff = page_cheeses_url_names.difference(set(all_cheeses_url_names))
            # if we request a page number that doesn't provide new cheeses, the site returns the
            # same page as before, and we know we can go to the next letter
            if not diff:
                keep_going = False
            else:
                all_cheeses_url_names.update(diff)


    return sorted(list(all_cheeses_url_names))
