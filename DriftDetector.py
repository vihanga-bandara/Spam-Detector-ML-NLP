#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:52:53 2019

@author: vihanga123
"""
import TwitterAPI
import preprocessing
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd

twitter = TwitterAPI.TwitterAPI()
preprocessor = preprocessing.preprocessing()
# authenitcate the twitter object before using API
twitter.authenticate()
# retrieve random tweet - know spam tweet for now
tweetObj = twitter.getTweet("EmptyForNow")
tweet = preprocessor.preprocess_tweet(tweetObj.text)
# tokenize the words using the best tokenizer available
tweet_tokens = word_tokenize(tweet)

# load the dataset   
df = pd.read_csv('SpamTweetsFinalDataset.csv', header=None)

# print general information about the dataset that is loaded
print(df.info())
print(df[1])

# check the class balance ratio / distribution
columnNames = list(df.head(0))
spam_tweets = df.loc[df[1] == 'spam']
spam_tweets = preprocessor.preprocess_spam_tweets(spam_tweets[0])
print(spam_tweets)
# get the spam only dataset and tokenize or use tfidf or fast text and get the most important words
# tokenize those words and add it to a list

# two functions are needed here. 
# two functions are made that would take the retreived random tweet tokens[random_tweet_tokens] and tokenized spam words [spam_tokens] with higher weights

# function1  
# It will call the vocab API and get for each [random_tweet_tokens] all similiar words 
# for each similiar word it will compare with each [spam_token] 
# if one or more comes up positive it will add a counter and then we can decide to end it or not
# then for each token that has a similarity will be noted and then it will be divided with the total number of words in the [random_tweet_tokens]
# if this score is better than 0.2 we will consider it as spam 

# function2
# if the above method does not prove to be useful then it will come to this method
# it will the vocab API and get for each [spam_token] all similiar words
# for each similiar word it will compare with each [random_tweet_token]
# if one or more comes up positive it will add a counter and then we can decide to end it or not
# we will compare the spam token and the word further and if it is the same
# it will be taken as a spam tweet

# both functions will finally send out if the tweet is spam or not. maybe a bool
# if this doesnt work out. It will try to use our unsupervised model to check whether it is in the correct cluster. 

# else the tweet will be shown and it will have the ability to be reported manually and then labelled accordingly by the admin

# if it does work out and if the tweet is recognised as spam using either function 1 or function 2 or unsupervised model
# we will add that tweet to our dataset and retrain the model.

# manually reported tweets will have the ability to be checked by user and then manually labelled. When the detector accuracy reduces by 80%
# it will take all the reported tweets that are labelled spam and then add it to dataset and retrained

# have an option where you could give tweets randomly and allow users to label them. If it contains tweets which arent really spam now.
# if many people report it as not spam. Those data will be removed from the dataset to make the retraining faster and efficient
# and make the model accuracy higher and precise

# retrieve spam user account detail by using tweet handle
# for now using a known spam account to retrieve the data

# since api is already initialised and authenticated retrieve user object details
twitterUser = twitter.findTweetUser(tweetObj)
userObj = twitter.getUser(twitterUser)
print(userObj)

# implementation for word2vec advanced fasttext
from gensim.models import Word2Vec

model_ted = Word2Vec(sentences=sentences_ted, size=100, window=2, min_count=3, workers=4, sg=0)
print(model_ted)

words = list(model_ted.wv.vocab)

print(model_ted.wv.most_similar('follower'))
