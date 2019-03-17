import pandas as pd
import numpy as np
import nltk
import pickle
import TwitterAuthentication
from nltk.tokenize import word_tokenize



# load the SpamTweetDetectModel word_features from directory
filename = "wordfeatures.p"
word_features = pickle.load(open(filename, 'rb'))

# load the SpamTweetDetectModel from directory
filename = "SpamTweetDetectModel.sav"
nltk_ensemble = pickle.load(open(filename, 'rb'))


# pre-process incoming data

def preprocess(tweet):
    # using regex to identify different combinations in the tweet

    # replacing email addresses with 'emailaddr'
    processed = tweet.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

    # replacing links / web addresses with 'webaddr'
    processed = processed.replace(
        r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', 'webaddr')

    # replacing money symbol with 'moneysymb'
    processed = processed.replace(r'$', 'moneysymbol')

    # replacing normal numbers with numbers
    # some not working might need to manually remove
    processed = processed.replace(r'^(\+|-)?\d+$', 'numbr')

    # remove punctuation
    processed = processed.replace(r'[^\w\d\s]', ' ')

    # replaces whitespaces between terms with single space
    processed = processed.replace(r'\s+', ' ')

    # removing leading and trailing whitespaces
    processed = processed.replace(r'^s+|\s+?$', '')

    # try hashtagsss as well which can be a new feature

    # change all letters to lowercase
    processed = processed.lower()

    # remove stop words or useless meaningless words from the tweets

    # from nltk.corpus import stopwords
    #
    # stop_words = set(stopwords.words('english'))

    # processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    #
    # # using a Porter stemmer to remove word stems
    # ps = nltk.PorterStemmer()
    # processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
    return processed


# define a find features function
def find_features(tweet):
    words = word_tokenize(tweet)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# retrieve random tweet
# for now using a known spam account to retrieve tweet

# create and initialise Twitter API object
TwitterAPI = TwitterAuthentication.TwitterAuthentication()

# authenitcate the twitter object before using API
TwitterAPI.authenticate()

# retrieve random tweet - know spam tweet for now
tweet = TwitterAPI.getRandomTweet("EmptyForNow")

tweet = preprocess(tweet.text)

features_test = find_features(tweet)
prediction_test = nltk_ensemble.classify(features_test)
