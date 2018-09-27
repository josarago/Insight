import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from collections import Counter
import string

regexp_tokenizer = RegexpTokenizer(r'\w+')

split_columns = ['aroma','flavour','rind','texture','type','ingredients']

ingredients = ['pasteurized','unpasteurized','buffalo','camel','cow',
     'donkey','goat','mare','moose','buffalo','sheep','water buffalo',
     'reindeer','yak'
    ]

df_columns = ['title','description','variety','region_1','country','price','points']


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
    def __init__(self,file='original'):
        if file=='original':
            filename="winemag-data-130k-v2.csv"
        elif file=='cleaned':
            filename="winemag_cleaned.csv"
        self.df = pd.read_csv(filename, index_col=0)
        self.train_df = None
        self.tests_df = None
        self.stop_words = StopWords('wine')
        self.variety_cnt = None
        self.descriptors = []
        with open("wiki_wine_descriptors.txt","r") as f:
            for line in f:
                self.descriptors.append(line.lower().strip('\n'))
        self.doc2vec_model = None

    def clean_df(self,remove_outliers_percent=4):
        """
            we're gonna use for sure the columns:'title','description','variety','region_1','country','price','points'
            So let's just get rid of any row that has a nan in any of these
        """
        sub_df = self.df.filter(df_columns)
        self.df = self.df[~sub_df.isna().any(axis=1)]
        # remove description that are too long or too short
        min_percentile = np.percentile(self.df.description.str.len().tolist(),remove_outliers_percent/2)
        max_percentile = np.percentile(self.df.description.str.len().tolist(),100-remove_outliers_percent/2)
        self.df = self.df[(min_percentile<self.df.description.str.len()) & (self.df.description.str.len()<max_percentile)]
        if os.path.exists('winemag_cleaned.csv'):
            print("'winemag_cleaned.csv' already exists and will be overwritten")
        self.df.to_csv('winemag_cleaned.csv')

    def get_region_variety_stop_words(self,save=True):# to exclude region related words from description
        file_name = "regions_varieties.stop_words"
        if not os.path.exists(file_name):
            region_stop=[]
            all_regions = list(set(self.df.region_1))
            [region_stop.extend(self.tokenize(region,use_stop='base')) for region in all_regions]

            # to exclude variety related words from description
            variety_stop=[]
            all_varieties = list(set(self.df.variety))
            [variety_stop.extend(self.tokenize(variety,use_stop='base')) for variety in all_varieties]
            print('done')
            extra_stop = list(set(region_stop+variety_stop))
            with open("regions_varieties.stop_words", 'wb') as fp:
                    pickle.dump(extra_stop, fp)
        else:
            with open (file_name, 'rb') as fp:
                extra_stop = pickle.load(fp)
                print("loading regions and varities stop words from 'regions_varieties.stop_words'")
        self.stop_words.set.update(extra_stop)
        self.stop_words.save()

    def tokenize(self,sentence,use_stop='all',**kwargs):
        vocab = kwargs.pop('vocab',None)
        if use_stop=='all':
            stop = self.stop_words.all
        elif use_stop=='base':
            stop = self.stop_words.base
        extra_stop = kwargs.pop('extra_stop',None)
        if extra_stop:
            stop.union(extra_stop)
        pretokenized = regexp_tokenizer.tokenize(sentence.lower())
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

    def get_wines_from_desc(self,input_str,model,topn=100):
        new_desc = self.tokenize(input_str,vocab=list(model.wv.vocab.keys()))
        similar_docs = model.docvecs.most_similar([model.infer_vector(new_desc)],topn=topn)
        indexes = [x[0] for x in similar_docs]
        return indexes, new_desc
