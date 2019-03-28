#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:10:03 2019

@author: vihanga123
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# spam user dataset unsupervised classification using k-means clustering

data = pd.read_csv('training_data_2_csv_UTF.csv')

data_numerical = data.select_dtypes(include=[np.number])

# data_unsupervise_classification = data_numerical.drop(['id','bot', 'listed_count','favourites_count','statuses_count'], axis = 1)
data_unsupervise_classification = data_numerical.drop(['bot', 'listed_count', 'favourites_count', 'statuses_count'],
                                                      axis=1)

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_unsupervise_classification)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

y = data.iloc[:, -1]

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(data_unsupervise_classification)

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

print('Prediction')
X = vectorizer.transform(['get free twitter followers'])
predicted = model.predict(X)
print(predicted)
