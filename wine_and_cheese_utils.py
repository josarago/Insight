import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import pickle
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
import string




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

class StopWords:
    def __init__(
            self,
            category,
            file_path=None,
        ):
        self.base_words = stopwords.words('english') + list(string.punctuation) + [str(x) for x in range(1900,2050)] + [str(x) for x in range(0,100)]
        if file_path:
            with open (file_path, 'rb') as fp:
                self.set = pickle.load(fp)
        else:
            self.set = set()
        self.category = category

    @property
    def base(self):
        return set(self.base_words)

    @property
    def all(self):
        return self.set.union(self.base_words)

    def load(file_path):
        with open (file_path, 'rb') as fp:
            self.set = pickle.load(fp)

    def save(self):
        with open("{}.stop_words".format(self.category), 'wb') as fp:
            pickle.dump(self.set.difference(self.base_words), fp)


# define the class cheese
class WineList:
    """
        Class used to deal with the wines
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    def __init__(self,filename="winemag-data-130k-v2.csv"):
        self.df = pd.read_csv(filename, index_col=0)
        self.train_df = None
        self.tests_df = None
        self.stop_words=StopWords('wine',file_path='wine.stop_words')
        self.variety_cnt = None
        self.descriptors = []
        with open("wiki_wine_descriptors.txt","r") as f:
            for line in f:
                self.descriptors.append(line.lower().strip('\n'))

    def tokenizer(self,sentence,use_stop='all',**kwargs):
        vocab = kwargs.pop('vocab',None)
        if use_stop=='all':
            stop = self.stop_words.all
        elif use_stop=='base':
            stop = self.stop_words.base
        extra_stop = kwargs.pop('extra_stop',None)
        if extra_stop:
            stop.union(extra_stop)
        pretokenized = word_tokenize(sentence.lower())
        if pretokenized:
            if vocab:
                return [i for i in pretokenized if i in vocab and i not in stop]
            else:
                return [i for i in pretokenized if i not in stop]
        else:
            return []

    def split_df(self,df,train_frac):
        # split the data frame
        msk = np.random.rand(len(df)) < train_frac
        train_df = df[msk]
        test_df = df[~msk]
        return train_df, test_df

    def get_variety_cnt(self):
        if self.variety_cnt==None:
            self.variety_cnt = {variety:len(self.df[self.df.variety==variety]) for variety in list(set(self.df.variety))}
            self.variety_cnt.pop(np.nan, None)

    def top_varieties(self,n_min=500, exclude=['blend','portuguese']):
        return {k:v for k,v in dict(self.variety_cnt).items() if v>n_min}

    def cat_desc_by_var(self,df):
        print("only using the top {} varieties".format(len(self.top_varieties())))
        return [df[df.variety==variety].description.str.cat() for variety in list(self.top_varieties().keys())]

    def get_tfidf(self,
            train_frac=0.6,
            ):
        # split the dataframe into training and test set
        self.split_df(train_frac)
        # get the tfidf object
        tfidf = TfidfVectorizer(
            sublinear_tf=True,
            min_df=5,
            norm='l2',
            encoding='latin-1',
        #     ngram_range=(1),# 2),
            stop_words=self.stop_words.all)

        features = tfidf.fit_transform(

            ).toarray()
        labels = list(top_varieties.keys())
        return tfidf

    def get_counter(self,input_list,tokenize=False,use_stop='all',n=300):
        if tokenize:
            cntr = Counter(self.tokenizer(input_list,use_stop=use_stop))
        else:
            cntr = Counter(input_list)
        return cntr

    def get_most_frequent(self,column_name,cum_sum_frac=0.6,**kwargs):
        """
            returns a list of the most frequently occuring items in the column 'column_name'
            (returns only a the ones that contribute to a max of cum_sum_frac of the total number of items)
        """
        # count the number of occurences of each value in the requested column
        count_df = self.df[column_name].value_counts()
        # figure how many items should be collected to make up for cum_sum_frac of the total
        n_items = np.round(cum_sum_frac*len(self.df))
        subset = count_df[count_df.cumsum()<np.round(cum_sum_frac*len(self.df))]
        percentage = sum(subset)/len(self.df)*100
        return subset, percentage

    def wine_str(self,this_wine):
        """
            nicer string representation of a single wine
        """
        returned_list =  [this_wine.title.values[0]]
        returned_list.append("Variety: {}  / Region: {} ".format(this_wine.variety.values[0],this_wine.region_1.values[0]))
        returned_list.append(this_wine.description.values[0])
        return returned_list
