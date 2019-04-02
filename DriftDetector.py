#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:52:53 2019

@author: vihanga123
"""
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import TwitterAPI
import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import json

twitter = TwitterAPI.TwitterAPI()
preprocessor = preprocessing.preprocessing()
# authenitcate the twitter object before using API
twitter.authenticate()
# retrieve random tweet - know spam tweet for now
tweetObj = twitter.getTweet("EmptyForNow")
tweet = preprocessor.preprocess_tweet(tweetObj.text)
# tokenize the words using the best tokenizer available
tweet_tokens = word_tokenize(tweet)

# get the spam only dataset and tokenize or use tfidf or fast text and get the most important words
# tokenize those words and add it to a list
# load the dataset   
df = pd.read_csv('SpamTweetsFinalDataset.csv', header=None)

# print general information about the dataset that is loaded
print(df.info())
print(df[1])

# check the class balance ratio / distribution
columnNames = list(df.head(0))
# using only the spam labelled data
spam_tweets = df.loc[df[1] == 'spam']
spam_tweets = preprocessor.preprocess_spam_tweets(spam_tweets[0])


# dummy function to avoid preprocessing and tokenization in tfid
def dummy_fun(doc):
    return doc


# initialize TF-ID vectorizer
tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)

tfidf_result = tfidf.fit_transform(spam_tweets)
# print features using the document matrix
print(tfidf.get_feature_names())


# a function to
def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    spam_tokens = []
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))
        if type(item[0]) == str:
            spam_tokens.append(item[0])
            print('adding spam token - {0}'.format(item[0]))

    return spam_tokens


spam_tokens = display_scores(tfidf, tfidf_result)
print(spam_tokens)
print(tweet_tokens)


def tweet_token_analogy_alg(tweet_tokens, spam_tokens):
    # function1
    # It will call the vocab API and get for each [random_tweet_tokens] all similiar words
    # for each similiar word it will compare with each [spam_token]
    # if one or more comes up positive it will add a counter and then we can decide to end it or not
    # then for each token that has a similarity will be noted and then it will be divided with the total number of words in the [random_tweet_tokens]
    # if this score is better than 0.2 we will consider it as spam
    # For each token get its associative words which have a score of above 70
    for token in tweet_tokens:

        # response = requests.get("https://twinword-word-associations-v1.p.rapidapi.com/associations/?entry=token",
        #                 headers={
        #                         "X-RapidAPI-Key": "5557d82ac7msheb9fc00a6b39b02p1f5141jsn4a6fd56d88ce"
        #                     }
        #                     )
        # print(response)
        #
        # with open('response.json', 'w') as outfile:
        #     json.dump(response.text, outfile)

        with open('response.json') as json_file:
            data = json.load(json_file)

        # get associative words for each token
        similiar_tokens_json = json.loads(data)
        print(similiar_tokens_json)
        similiar_tokens = list(similiar_tokens_json['associations_scored'])
        for simtoken in similiar_tokens:
            for sptoken in spam_tokens:

        # words1 = sentence1.split(' ')
        # words2 = sentence2.split(' ')
        #
        # # The meaning of the sentence can be interpreted as the average of its words
        # sentence1_meaning = word2vec(words1[0])
        # count = 1
        # for w in words1[1:]:
        #     sentence1_meaning = np.add(sentence1_meaning, word2vec(w))
        #     count += 1
        # sentence1_meaning /= count
        #
        # sentence2_meaning = word2vec(words2[0])
        # count = 1
        # for w in words2[1:]:
        #     sentence2_meaning = np.add(sentence2_meaning, word2vec(w))
        #     count += 1
        # sentence2_meaning /= count
        #
        # # Similarity is the cosine between the vectors
        # similarity = np.dot(sentence1_meaning, sentence2_meaning) / (
        #         np.linalg.norm(sentence1_meaning) * np.linalg.norm(sentence2_meaning))


# run func1
tweet_token_analogy_alg(tweet_tokens, spam_tokens)
# two functions are needed here. 
# two functions are needed that would take the retreived random tweet tokens[random_tweet_tokens] and tokenized spam words [spam_tokens] with higher weights


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
# twitterUser = twitter.findTweetUser(tweetObj)
# userObj = twitter.getUser(twitterUser)
# print(userObj)
#
# # implementation for word2vec advanced fasttext
# from gensim.models import Word2Vec
#
# model_ted = Word2Vec(sentences=sentences_ted, size=100, window=2, min_count=3, workers=4, sg=0)
# print(model_ted)
#
# words = list(model_ted.wv.vocab)
#
# print(model_ted.wv.most_similar('follower'))


# Count Vectorizer Implementation
# from sklearn.feature_extraction.text import CountVectorizer
#
# # list of text documents
# text = ["The quick brown fox jumped over the lazy dog."]
# # create the transform
# vectorizer = CountVectorizer()
# # tokenize and build vocab
# vectorizer.fit(text)
# # summarize
# print(vectorizer.vocabulary_)
# # encode document
# vector = vectorizer.transform(text)
# # summarize encoded vector
# print(vector.shape)
# print(type(vector))
# print(vector.toarray())
#
# print(vectorizer.vocabulary_)
#
# # encode another document
# text2 = ["doggos the"]
# vector = vectorizer.transform(text2)
# print(vector.toarray())
