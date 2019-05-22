import sys
import sklearn
import pandas as pd
import numpy as np
import nltk
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


class TweetDetectModel:
    tweets_dataset = None
    tweets_class_col = None
    tweets_col = None
    tweets_processed_col = None

    def get_package_versions(self):
        package_version = dict()
        package_version["Python"] = 'Python: {}', format(sys.version)
        package_version["Python"] = 'NLTK: {}', format(nltk.__version__)
        package_version["Python"] = 'Scikit-learn: {}', format(sklearn.__version__)
        package_version["Python"] = 'Pandas: {}', format(pd.__version__)
        package_version["Python"] = 'Numpy: {}', format(np.__version__)
        package_version["Python"] = 'MatPlotLib: {}', format(matplotlib.__version__)
        return package_version

    def load_dataset(self):
        # try to load the dataset
        try:
            df = pd.read_csv('dataset/SpamTweetsFinalDataset.csv', header=None)
            print(df.info())
            print(df.head())

            # check the class balance ratio / distribution
            df_spam = df.loc[df[1] == 'spam']
            df_ham = df.loc[df[1] == 'ham']
            ratio = int(df.count()[0] / df_spam.count()[0])
            print("ratio of spam to to total data is {0}".format(ratio))

            # separate label column to add numerical label
            column_names = list(df.head(0))
            classes_column = df[column_names[1]].str.strip()
            tweets_column = df[column_names[0]].str.strip()

            # store dataset in object
            self.tweets_dataset = df
            # store class column in object
            self.tweets_class_col = classes_column
            # store tweet column in object
            self.tweets_col = tweets_column

            return True

        except Exception as e:
            # if loading dataset does not work return false
            print("Error loading the dataset")

            return False

    def preprocessing_tweets(self, unprocessed_tweets):

        # pre-processing the tweets before classification
        # using regex to identify different combinations in the tweet

        # replacing email addresses with 'emailaddr'
        processed = unprocessed_tweets.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

        # replacing links / web addresses with 'webaddr'
        processed = processed.str.replace(
            r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', 'webaddr')

        # replacing money symbol with 'moneysymb'
        processed = processed.str.replace(r'$', 'moneysymbol')

        # replacing normal numbers with numbers
        # some not working might need to manually remove
        processed = processed.str.replace(r'^(\+|-)?\d+$', 'numbr')

        # remove punctuation
        processed = processed.str.replace(r'[^\w\d\s]', ' ')

        # replaces whitespaces between terms with single space
        processed = processed.str.replace(r'\s+', ' ')

        # removing leading and trailing whitespaces
        processed = processed.str.replace(r'^s+|\s+?$', '')

        # try hashtagsss as well which can be a new feature

        # change all letters to lowercase
        processed = processed.str.lower()

        # remove stop words or useless meaningless words from the tweets

        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))

        processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

        # using a Porter stemmer to remove word stems
        ps = nltk.PorterStemmer()
        processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

        self.tweets_processed_col = processed
        return processed

    def label_encoding(self, classes_column):
        # convert the labels into binary values
        # where 0 = ham and 1 = spam
        label_encoder = LabelEncoder()
        self.tweets_class_col = label_encoder.fit_transform(classes_column)

    def main(self):
        # get package version details
        package_version = self.__get_package_versions()

        # load dataset
        if self.load_dataset():
            # encode classes using LabelEncoder
            self.label_encoding(self.tweets_class_col)
            unprocessed_tweets = self.tweets_col
            self.preprocessing_tweets(unprocessed_tweets)
            # finalising dataset with labelled classes and processed tweets
            self.tweets_dataset[1] = self.tweets_class_col
            self.tweets_dataset[0] = self.tweets_processed_col
