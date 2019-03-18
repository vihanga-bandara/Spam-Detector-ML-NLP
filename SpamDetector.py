import pandas as pd
import numpy as np
import nltk
import pickle
import TwitterAuthentication
import re
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


    # replacing money symbol with 'moneysymb'
    processed = processed.replace(r'$', 'moneysymbol')

    # replacing normal numbers with numbers
    # some not working might need to manually remove
    processed = processed.replace(r'^(\+|-)?\d+$', 'numbr')

    # replaces whitespaces between terms with single space
    processed = processed.replace(r'\s+', ' ')

    # removing leading and trailing whitespaces
    processed = processed.replace(r'^s+|\s+?$', '')

    # try hashtagsss as well which can be a new feature
    # remove hashtags for now
    processed = processed.replace('#', '')

    # remove @handle mentioning names in the tweet
    processed = re.sub(r'@[A-Za-z0-9]+', '', processed, flags=re.MULTILINE)

    # change all letters to lowercase
    processed = processed.lower()

    # remove ASCII character
    #     initial_str = 'Some text ðŸ‘‰ðŸ‘ŒðŸ’¦âœ¨ and some more text'
    #     clean_str = ''.join([c for c in initial_str if ord(c) < 128])
    #     print(clean_str)  # Some text  and some more text

    # remove specific character from the tweet
    cleansed = False
    while not cleansed:
        if '\n' in processed:
            print("Removing next line from words")
            processed = processed.replace("\n", " ")

        elif "-&gt;" in processed:
            print("Removing special characters from words")
            processed = processed.replace("-&gt;", " ")

        elif "&lt;-" in processed:
            print("Removing special characters form words")
            processed = processed.replace("&lt;-", " ")

        else:
            print("Formatting done")
            cleansed = True
    processed = processed

    # replacing links / web addresses with 'webaddr'
    # processed = processed.replace(
    #     r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'webaddr')

    processed = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'webaddr',
                       processed, flags=re.MULTILINE)

    # remove punctuation
    # processed = processed.replace(r'[^\w\d\s]', ' ')
    processed = re.sub(r'[^\w\d\s]', '', processed, flags=re.MULTILINE)

    # remove stop words or useless meaningless words from the tweets
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(processed)
    processed = [word for word in tokens if word not in stop_words]
    tweet = " ".join(processed)
    #
    # # using a Porter stemmer to remove word stems
    # ps = nltk.PorterStemmer()
    # processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
    return tweet


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
print(prediction_test)
