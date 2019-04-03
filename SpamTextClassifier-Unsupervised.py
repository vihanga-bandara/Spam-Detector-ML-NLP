#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:10:03 2019

@author: vihanga123
"""

# spam tweet dataset unsupervised classification using tf-idk and k-means clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv('SpamTweetsFinalDataset.csv', header=None)

# check the class balance ratio / distribution
columnNames = list(df.head(0))
class_y = df[columnNames[1]].str.strip()
class_X = list(df[columnNames[0]].str.strip())

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(class_X)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print('Cluster')
    for ind in order_centroids[i, :10]:
        print(terms[ind])

import pickle

# save vectorizer using pickle
filename = "Unsupervised_Vectorizer_TFIDF.p"
pickle.dump(vectorizer, open(filename, "wb"))

# save unsupervised model using pickle
filename = 'Unsupervised_KMeans_Model.sav'
pickle.dump(model, open(filename, 'wb'))

X = vectorizer.transform(['Free stuff only for the first 100 twitter followers'])
predicted = model.predict(X)
print(predicted)
