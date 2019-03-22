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
df = pd.read_csv('SpamTweetsFinalDataset.csv', header=None)

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
print(classes[:10])
print(Y[:10])
# convert the class labels using onehotencoder
# from sklearn.preprocessing import OneHotEncoder
# binaryEncoder = OneHotEncoder()
# X = binaryEncoder.fit_transform(classes).toarray()

# store the twitter data
tweets = df[0].str.strip()

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

# use the 250 most common words as features
word_features = list(all_words.keys())[:250]


#define a find features function
def find_features(tweet):
    words = word_tokenize(tweet)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# example
features = find_features(processed[65])
for key, value in features.items():
    if value == True:
        print(key)

# find features for all tweets
tweets = zip(processed, Y)
tweetsList = list(tweets)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(tweetsList)

# call find features function for each tweet
featuresets = [(find_features(text), label) for (text, label) in tweetsList]

# split training and testing data sets using sklearn
from sklearn import model_selection

training, testing = model_selection.train_test_split(featuresets, test_size=0.2, random_state=seed)
print('Training: {}'.format(len(training)))
print('Testing: {}'.format(len(testing)))

# classification training using scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# classifaction tools and matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.svm.classes import OneClassSVM
# from sklearn.ensemble.voting_classifier import VotingClassifier
# from sklearn.ensemble.weight_boosting import AdaBoostClassifier
# from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
# from sklearn.ensemble.bagging import BaggingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import LinearSVC
# from sklearn.mixture import GaussianMixture

# wrap classification model  in NLTK
from nltk.classify.scikitlearn import SklearnClassifier

# DecisionTreeClassifier 
nltk_model_decision_tree = SklearnClassifier(DecisionTreeClassifier())
nltk_model_decision_tree.train(training)
accuracy_training = nltk.classify.accuracy(nltk_model_decision_tree, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_model_decision_tree, testing) * 100

print('## DecisionTree Classifier ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# KNeighborsClassifier 
nltk_model_decision_tree = SklearnClassifier(KNeighborsClassifier())
nltk_model_decision_tree.train(training)
accuracy_training = nltk.classify.accuracy(nltk_model_decision_tree, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_model_decision_tree, testing) * 100

print('## KNeighborsClassifier ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# RandomForestClassifier 
nltk_model_decision_tree = SklearnClassifier(RandomForestClassifier())
nltk_model_decision_tree.train(training)
accuracy_training = nltk.classify.accuracy(nltk_model_decision_tree, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_model_decision_tree, testing) * 100

print('## RandomForestClassifier ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# LogisticRegression 
nltk_model_decision_tree = SklearnClassifier(LogisticRegression())
nltk_model_decision_tree.train(training)
accuracy_training = nltk.classify.accuracy(nltk_model_decision_tree, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_model_decision_tree, testing) * 100

print('## LogisticRegression ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# SVM Linear 
nltk_model_decision_tree = SklearnClassifier(SVC())
nltk_model_decision_tree.train(training)
accuracy_training = nltk.classify.accuracy(nltk_model_decision_tree, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_model_decision_tree, testing) * 100

print('## SVM Linear ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# NaiveBayes 
nltk_model_decision_tree = SklearnClassifier(MultinomialNB())
nltk_model_decision_tree.train(training)
accuracy_training = nltk.classify.accuracy(nltk_model_decision_tree, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_model_decision_tree, testing) * 100

print('## NaiveBayes ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# SGDClassifier 
nltk_model_decision_tree = SklearnClassifier(SGDClassifier())
nltk_model_decision_tree.train(training)
accuracy_training = nltk.classify.accuracy(nltk_model_decision_tree, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_model_decision_tree, testing) * 100

print('## SGDClassifier ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# ensemble method - Voting Classifier
from sklearn.ensemble import VotingClassifier

# Defining the model that will be trained
names = ['Decision Tree', 'Random Forest', 'Logistic Regression']

classifier = [DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]

models = zip(names, classifier)
models_list = list(models)

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models_list, voting='hard', n_jobs=1))
nltk_ensemble.train(training)
accuracy_training = nltk.classify.accuracy(nltk_ensemble, training) * 100
accuracy_testing = nltk.classify.accuracy(nltk_ensemble, testing) * 100

print('## Ensemble Method Voting Classifier Accuracy ##')
print('Accuracy_Training: {}'.format(accuracy_training))
print('Accuracy_Testing: {}'.format(accuracy_testing))
print('####')

# making class label prediction for testing set
text_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(text_features)

# print a classification report and a confusion matrix
print(classification_report(labels, prediction))
new = text_features[1]
# pd.DataFrame(
#    confusion_matrix(labels, prediction),
#    index=[['actual', 'actual'], ['ham', 'spam']],
#    columns=[['predicted', 'predicted'], ['ham', 'spam']])

test_tweet_value = input("enter sentence: ")
test_tweet_value = test_tweet_value.lower()
features_test = find_features(test_tweet_value)
prediction_test = nltk_ensemble.classify(features_test)
print(prediction_test)

import pickle

# save wordfeatures using pickle
filename = "wordfeatures.p"
pickle.dump(word_features, open(filename, "wb"))

# save model using pickle
filename = 'SpamTweetDetectModel.sav'
pickle.dump(nltk_ensemble, open(filename, 'wb'))
