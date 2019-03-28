#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:52:53 2019

@author: vihanga123
"""
import TwitterAPI
import preprocessing

twitter = TwitterAPI.TwitterAPI()
preprocessor = preprocessing.preprocessing()
# authenitcate the twitter object before using API
twitter.authenticate()

# retrieve random tweet - know spam tweet for now
tweetObj = twitter.getTweet("EmptyForNow")

tweet = preprocessor.preprocess(tweetObj.text)

features_test = find_features(tweet)
prediction_test = nltk_ensemble.classify(features_test)
print(prediction_test)

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
