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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import re

print('Python: {}', format(sys.version))
print('NLTK: {}', format(nltk.__version__))
print('Scikit-learn: {}', format(sklearn.__version__))
print('Pandas: {}', format(pandas.__version__))
print('Numpy: {}', format(numpy.__version__))

import pandas as pd
import numpy as np
import nltk

# load the dataset
df = pd.read_csv('dataset/NewSpamDataTesting.csv', header=None)

# print general information about the dataset that is loaded
print(df.info())
print(df.head())

# check the class balance ratio / distribution
columnNames = list(df.head(0))
classes = df[columnNames[1]].str.strip()

# pre-processing the data before classification

# convert the labels into binary values
# where 0 = ham and 1 = spam
from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()
df[1] = labelEncoder.fit_transform(df[1])

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

df[0] = processed

X = df[0]
y = df[1]
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm.classes import OneClassSVM
from sklearn.ensemble.voting_classifier import VotingClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

pipeline = Pipeline(
    [('vectorizer', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('classifier', LogisticRegression())])

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)

mean = scores.mean()
std = scores.std()
print(mean)
print(std)

print(pipeline.get_params())
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

input = ["get free money"]
train_score = pipeline.fit(X_train, y_train)
print(train_score)
y_pred = pipeline.predict(X_test)
print(np.mean(y_pred == y_test))
score = pipeline.predict(input)
print(score)
print(classification_report(y_test, y_pred))
# confusion_matrix = pd.DataFrame(
#     confusion_matrix(y_test, y_pred),
#     index=[['actual', 'actual '], ['ham', 'spam']],
#     columns=[['predicted', 'predicted'], ['ham', 'spam']])
# print(confusion_matrix)

cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import savefig

ax = plt.subplot()
svm = sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Greens')
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['ham', 'spam']);
ax.yaxis.set_ticklabels(['ham', 'spam']);
figure = svm.get_figure()
figure.savefig('images/confusion_matrix.png', dpi=400)
plt.close()
# predict probabilities
probs = pipeline.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')

pyplot.savefig('images/roc_curve.png', dpi=400)
pyplot.show()
pyplot.close()
print(confusion_matrix)

