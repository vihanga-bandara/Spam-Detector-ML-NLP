#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 02:48:29 2019

@author: vihanga123
"""

import sys
import nltk
import sklearn
import pandas
import numpy

print('Python: {}', format(sys.version))
print('NLTK: {}', format(nltk.__version__))
print('Scikit-learn: {}', format(sklearn.__version__))
print('Pandas: {}', format(pandas.__version__))
print('Numpy: {}', format(numpy.__version__))

import pandas as pd
import numpy as np
import nltk

# load the dataset   
df = pd.read_csv('SpamTweetsFinalDataset.csv')

# print general information about the dataset that is loaded
print(df.info())
print(df.head())

# check the class balance ratio / distribution
columnNames = list(df.head(0))
classes = df[columnNames[1]].str.strip()
print(classes.value_counts())

# pre-processing the data before classification

# convert the labels into binary values 
# where 0 = ham and 1 = spam
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(classes)

# convert the class labels using onehotencoder
# from sklearn.preprocessing import OneHotEncoder
# binaryEncoder = OneHotEncoder()
# X = binaryEncoder.fit_transform(classes).toarray()

# store the twitter data
tweets = df['Tweet'].str.strip()

# using regex to identify different combinations in the tweet

# replacing email addresses with 'emailaddr'
processed = tweets.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

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

#creating bag of words model
from nltk.tokenize import word_tokenize

all_words = []

for tweet in processed:
    words = word_tokenize(tweet)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

# print the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most Common words: {}'.format(all_words.most_common(500)))

# use the 1500 most common words as features
word_features = list(all_words.keys())[:500]


#define a find features function
def find_features(tweet):
    words = word_tokenize(tweet)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# example
# features = find_features(processed['Tweet'])
# for key,value in features.items():
#    if value == True:
#        print (key)

# find features for all tweets
tweets = zip(processed, Y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(tweets)

# call find features function for each tweet
featuresets = [(find_features(text), label) for (text, label) in tweets]

# split training and testing data sets using sklearn
from sklearn import model_selection

training, testing = model_selection.train_test_split(featuresets, test_size=0.2, random_state=seed)
