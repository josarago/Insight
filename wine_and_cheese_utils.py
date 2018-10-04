import os
import pandas as pd
import datetime
import numpy as np
import requests
from bs4 import BeautifulSoup
import warnings
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython.display import display, clear_output
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter
import string

regexp_tokenizer = RegexpTokenizer(r'\w+')

split_columns = ['aroma','flavour','rind','texture','type','ingredients']

ingredients = ['pasteurized','unpasteurized','buffalo','camel','cow',
     'donkey','goat','mare','moose','buffalo','sheep','water buffalo',
     'reindeer','yak'
    ]

df_columns = ['title','description','variety','region_1','country','price','points']

warnings.filterwarnings("ignore")


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

    def get_date_suffix(self):
        return datetime.datetime.now().strftime('%b_%d_%Y_%H-%M')

    def save(self,overwrite=False):
        filename =  "{}.stop_words".format(self.category)
        if os.path.exists(filename):
            if overwrite:
                print("{} already exists and will be overwritten".format(filename))
            else:
                filename =  "{}_{}.stop_words".format(self.category,self.get_date_suffix())
                print("file already exists, will be saved as {}".format(filename))
        with open(filename, 'wb') as fp:
            pickle.dump(self.set.difference(self.base_words), fp)

# define the class cheese
class WineList:
    """
        Class used to deal with the wines
    """
    def __init__(self,file='original'):
        if file=='original':
            filename="winemag-data-130k-v2.csv"
        elif file=='cleaned':
            filename="winemag_cleaned.csv"
        else:
            raise ValueError("file can only be 'original' or 'cleaned'")
        self.df = pd.read_csv(filename, index_col=0)
        self.train_df = None
        self.tests_df = None
        self.stop_words = StopWords('wine')
        self.descriptors = []
        self.column_cnt = dict.fromkeys(df_columns,None)
        with open("wiki_wine_descriptors.txt","r") as f:
            for line in f:
                self.descriptors.append(line.lower().strip('\n'))
        self.tagged_data = None
        self.tagged_data_set = None

    def get_date_suffix(self):
        return datetime.datetime.now().strftime('%b_%d_%Y_%H-%M')

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

    def get_tagged_data(self,recompute=False,file_name = "tagged_data_set.pkl"):
            """get TaggedDocument either from file or from dataframe"""

            self.tagged_data = [TaggedDocument(words=self.tokenize(row.description), tags=[row.Index]) for row in self.df.itertuples()]
            if recompute or not os.path.exists(file_name):
                self.tagged_data_set = dict(zip([x.tags[0] for x in self.tagged_data], [set(x.words) for x in self.tagged_data]))
                print("saving " + file_name)
                if os.path.exists(file_name):
                    file_name = "tagged_data_set_{}.pkl".format(self.get_date_suffix())
                    with open(file_name, 'wb') as f:
                        pickle.dump(self.tagged_data_set, f, pickle.HIGHEST_PROTOCOL)
            else:
                print("loading " + file_name)
                with open(file_name, 'rb') as f:
                    self.tagged_data_set = pickle.load(f)


    def add_region_variety_stop_words(self,save=True,overwrite=False):# to exclude region related words from description
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
                print("loading stop words from {}".format(file_name))
                extra_stop = pickle.load(fp)
        self.stop_words.set.update(extra_stop)
        if save:
            self.stop_words.save(overwrite=False)

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

    def get_doc2vec_model(
            self,
            retrain = False,
            from_file="doc2vec.model",
            max_epochs = 20,
            vec_size = 40,
            alpha = 0.025,
            dalpha = -0.0002,
            min_count=10,
            dm =0,
            dbow_words=1,
        ):
        # load or train the model
        if retrain or not os.path.exists(from_file):
            model_file_name = "doc2vec_{}.model".format(self.get_date_suffix())
            self.model = Doc2Vec(
                            vector_size=vec_size,
                            alpha=alpha,
                            min_count=min_count,
                            dm =dm,
                            dbow_words=dbow_words,
                        )
            if not self.tagged_data:
                raise ValueError("use 'get_tagged_data' method to create or load TaggedDocument object")
            self.model.build_vocab(self.tagged_data)
            for epoch in range(max_epochs):
                clear_output(wait=True)
                display('iteration {0}'.format(epoch))
                self.model.train(self.tagged_data,
                            total_examples=self.model.corpus_count,
                            epochs=self.model.epochs)
                # decrease the learning rate
                self.model.alpha+=dalpha
                # fix the learning rate, no decay
                self.model.min_alpha = self.model.alpha
            self.model.save(model_file_name)
        else:
            print("loading Doc2Vec model from {}".format(from_file))
            self.model = Doc2Vec.load(from_file)

    def split_df(self,df,train_frac):
        # split the data frame
        msk = np.random.rand(len(df)) < train_frac
        train_df = df[msk]
        test_df = df[~msk]
        return train_df, test_df

    # def get_column_cnt(self,column_name):
    #     if self.column_cnt[column_name]==None:
    #         self.column_cnt[column_name] = {key:{len(self.df[self.df[column_name]==key])} for key in list(set(self.df[column_name]))}
    #         self.column_cnt.pop(np.nan, None)

    def get_column_cnt_list(self,column_name):
        cnt = -1
        self.column_cnt[column_name] = []
        for key in list(set(self.df[column_name])):
            self.column_cnt[column_name].append({'name':key,'count':len(self.df[self.df[column_name]==key])})

    def get_top_keys(self,column_name,n_min=500):
        if  not self.column_cnt[column_name]:
            self.get_column_cnt_list(column_name)
        return [x for x in self.column_cnt[column_name] if x['count']>=n_min]

    # def get_top_keys(self,column_name,n_min=500):
    #     if  not self.column_cnt[column_name]:
    #         self.get_column_cnt(column_name)
    #     return {k:v for k,v in dict(self.column_cnt[column_name]).items() if v>n_min}

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

    def get_doc2vec_wines_from_desc(self,desc,topn=100):
        similar_docs = self.model.docvecs.most_similar([self.model.infer_vector(desc)],topn=topn)
        indexes = [x[0] for x in similar_docs]
        return indexes

    def is_in_column(self,token,column_name):
        """
            fast way to check if an input string appears in a column
        """
        token = token.lower()
        # lowered_column_str = [x.lower() for x in set(self.df[column_name])]
        return any([token in this_column_str.lower() for this_column_str in set(self.df[column_name])])

    def is_in_vocab(self,token,model):
        return token.lower() in model.wv.vocab

    def get_matched_columns(self,token,column_list):
        """
            find all the column in column list for which there is an exact match with a token
        """
        token = token.lower()
        columns_matched = set()
        [columns_matched.add(column_name) for column_name in column_list if self.is_in_column(token,column_name)]
        return columns_matched

    def get_match_dict(self,input_str,model):
        lookup_columns = 'variety','region_1'
        desc = self.tokenize(input_str)
        match_dict = {
                'description':[],
                'variety':[],
                'region_1':[],
            }
        for token in desc:
            [match_dict[column_name].append(token) for column_name in lookup_columns if self.is_in_column(token,column_name)]
            if self.is_in_vocab(token,model):
                match_dict['description'].append(token)
        return match_dict, desc

    def get_description_match_series_from_sets(self,desc):
        set_desc = set(desc)
        indexes = index=[x.tags[0] for x in self.tagged_data]
        data=[set_desc.issubset(this_set) for this_set in self.tagged_data_set.values()]
        return pd.Series(data=data,index=indexes)

    def get_exact_match_from_description(self,desc):
        set_desc = set(desc)
        indexes = [index for _, (index, this_desc) in enumerate(self.tagged_data_set.items()) if set_desc.issubset(this_desc)]
        return indexes
