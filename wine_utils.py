import os
import pandas as pd
import datetime
import numpy as np
import requests
from bs4 import BeautifulSoup
import warnings
import pickle
from collections import Counter
import string
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython.display import display, clear_output
from scipy.spatial.distance import cosine
from scipy.stats import gaussian_kde

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer

SPLIT_COLUMNS = ['aroma','flavour','rind','texture','type','ingredients']

DF_COLUMNS = ['title','description','variety','region_1','country','price','points']

# Seems like 0.9999 is a VERY conservative threshold (see Jupyter Notebook)
COSINE_SIMILARITY_THRESHOLD = 0.9999

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
            fp.close()
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
        fp.close()

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
        self.column_cnt = dict.fromkeys(DF_COLUMNS,None)
        with open("wiki_wine_descriptors.txt","r") as f:
            for line in f:
                self.descriptors.append(line.lower().strip('\n'))
        self.tagged_data = None
        self.tagged_data_set = None
        self.mean_vects_dict = None

    def clean_df(self,remove_outliers_percent=4,wine_csv_file='winemag_cleaned.csv'):
        """
            we're gonna use for sure the columns:'title','description','variety','region_1','country','price','points'
            So let's just get rid of any row that has a nan in any of these
        """
        column_filtered_df = self.df.filter(DF_COLUMNS)
        self.df = self.df[~column_filtered_df.isna().any(axis=1)]

        # remove description that are too long or too short
        min_percentile = np.percentile(self.df.description.str.len().tolist(),remove_outliers_percent/2)
        max_percentile = np.percentile(self.df.description.str.len().tolist(),100-remove_outliers_percent/2)
        self.df = self.df[(min_percentile<self.df.description.str.len()) & (self.df.description.str.len()<max_percentile)]
        if os.path.exists(wine_csv_file):
            print("{} already exists and will be overwritten".format(wine_csv_file))
        self.df.to_csv(wine_csv_file)

    def get_tagged_data(self,recompute=False,file_name="tagged_data_set.pkl"):
            """get TaggedDocument either from file or from dataframe"""
            self.tagged_data = [TaggedDocument(words=self.tokenize(row.description),tags=[row.Index]) for row in self.df.itertuples()]

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
            fp.close()
        self.stop_words.set.update(extra_stop)
        if save:
            self.stop_words.save(overwrite=False)

    def tokenize(self,sentence,use_stop='all',**kwargs):
        regexp_tokenizer = RegexpTokenizer(r'\w+')
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

    def get_mean_region_vect_dict(self,n_min=50,recompute=False):
        self.top_regions_list = self.get_top_keys('region_1',n_min=n_min)
        if os.path.exists("mean_region_docvecs_dict.pkl") and not recompute:
            with open ("mean_region_docvecs_dict.pkl", 'rb') as fp:
                self.mean_vects_dict =  pickle.load(fp)
            fp.close()
        else:
            # initialize a dictionary to store the mean vector for each region
            self.mean_vects_dict = dict.fromkeys([x['name'] for x in self.top_regions_list],np.zeros(self.model.vector_size))
            # for every region in the list
            for region in self.top_regions_list:
                # get the rows for wines of that region
                region_df = self.df[self.df.region_1==region['name']]
                # shuffle the filtered DataFrame wines
                region_df = region_df.sample(frac=1)
                # initialize a counter
                cnt=0
                # and a "previous vector"
                prev_vec = np.ones(self.model.vector_size).reshape(1, -1)
                keepgoing = True

                for row in region_df.itertuples():
                    if keepgoing:
                        cnt+=1
                        # infer the feature vector
                        new_vec = self.model.infer_vector(self.tokenize(row.description)).reshape(1, -1)
                        # update the mean vector
                        self.mean_vects_dict[region['name']] = ((cnt-1)*self.mean_vects_dict[region['name']]+new_vec)/cnt
                        # update the previous vector
                        prev_vec = self.mean_vects_dict[region['name']].copy()
                        # test if we reached the threshold for the convergence of the mean vector
                        keepgoing = cosine_similarity(prev_vec,self.mean_vects_dict[region['name']])<COSINE_SIMILARITY_THRESHOLD
                    else:
                        break

    def get_kde_argmax(self,n_min=50):
        self.top_regions_list = self.get_top_keys('region_1',n_min=n_min)
        # initialize a dictionary to store the mean vector for each region
        self.kde_argmax_vects_dict = dict.fromkeys([x['name'] for x in self.top_regions_list],np.zeros(self.model.vector_size))
        # for every region in the list
        for region in self.top_regions_list:
            # get the rows for wines of that region
            region_df = self.df[self.df.region_1==region['name']]
            # shuffle the filtered DataFrame wines
            region_df = region_df.sample(n=n_min)
            region_mat = []
            for wine in region_df.itertuples():
                desc = self.tokenize(wine.description)
                region_mat.append(self.model.infer_vector(desc))
            region_mat = np.transpose(np.vstack(region_mat))
            kernel = gaussian_kde(region_mat)
            self.kde_argmax_vects_dict[region['name']] = region_mat.T[np.argmax(kernel)]


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

    def is_in_vocab(self,token):
        return token.lower() in self.model.wv.vocab

    def get_matched_columns(self,token,column_list):
        """
            find all the column in column list for which there is an exact match with a token
        """
        token = token.lower()
        columns_matched = set()
        [columns_matched.add(column_name) for column_name in column_list if self.is_in_column(token,column_name)]
        return columns_matched

    def get_match_dict(self,input_str):
        lookup_columns = 'variety','region_1'
        desc = self.tokenize(input_str)
        match_dict = {
                'description':[],
                'variety':[],
                'region_1':[],
            }
        for token in desc:
            [match_dict[column_name].append(token) for column_name in lookup_columns if self.is_in_column(token,column_name)]
            if self.is_in_vocab(token,self.model):
                match_dict['description'].append(token)
        return match_dict, desc

    def get_regional_archetype(self,region_name,topn=30,method='mean'):
        """
            a function to display the wine from a region most similar to the average feature vector of that region
        """
        if method=='mean':
            archetype_vect = np.array(self.mean_vects_dict[region_name])
        elif method=='kde_mode':
            archetype_vect = np.array([self.kde_argmax_vects_dict[region_name]])
        # find the most similar docsvec for its mean vector
        similar_docs = self.model.docvecs.most_similar(archetype_vect,topn=topn)
        keepgoing = True
        indx = -1
        return_str = None
        while keepgoing:
            indx+=1
            if self.df.loc[similar_docs[indx][0]].region_1==region_name:
                keepgoing=False
                archetype_vect_wine = self.df.loc[similar_docs[indx][0]]
        return archetype_vect_wine

    def get_description_match_series_from_sets(self,desc):
        set_desc = set(desc)
        indexes = index=[x.tags[0] for x in self.tagged_data]
        data=[set_desc.issubset(this_set) for this_set in self.tagged_data_set.values()]
        return pd.Series(data=data,index=indexes)

    def get_exact_match_from_description(self,desc):
        set_desc = set(desc)
        indexes = [index for _, (index, this_desc) in enumerate(self.tagged_data_set.items()) if set_desc.issubset(this_desc)]
        return indexes

    def get_direct_region_wines(self,region_name,desc=None):
        """
            return indexes of wines from region 'region_name' and, if provided
            wines from region region_name with tokens from desc
        """
        region_indexes = self.df[self.df.region_1==region_name].index
        if desc:
            desc_indexes = self.get_exact_match_from_description(desc)
            region_indexes = list(set(region_indexes) & set(desc_indexes))
        return region_indexes

    def get_doc2vec_region_wines(
                self,
                region_name,
                include_indexes=[],
                exclude_indexes=[],
                topn=5,
                method='mean',
                desc=None,
                weight=.33
            ):
        """
            this function finds the topn most similar wines to the average wine from region_name
            and if a desc if provided, the most ssimilar
        """
        if region_name in [x['name'] for x in self.top_regions_list]:
            if method=='mean':
                wine_vect = self.mean_vects_dict[region_name].copy()
            elif method=='kde_mode':
                wine_vect = self.kde_argmax_vects_dict[region_name].copy()
            if desc:
                for token in desc:
                    wine_vect+= self.model[token]*weight
            similar_docs = self.model.docvecs.most_similar(wine_vect,topn=topn)
            indexes = [x[0] for x in similar_docs]
            if  len(exclude_indexes)>0:
                indexes = [x for x in indexes if x not in exclude_indexes]
            if len(include_indexes)>0:
                indexes = [x for x in indexes if x in include_indexes]
            return indexes, wine_vect
        return [], None

    def get_date_suffix(self):
        return datetime.datetime.now().strftime('%b_%d_%Y_%H-%M')
