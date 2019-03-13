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

# load the dataset   
df = pd.read_csv('SpamTweetsFinalDataset.csv')

# print general information about the dataset that is loaded
print(df.info())
print(df.head())

# check the class balance ratio / distribution
columnNames = list(df.head(0))
classes = df[columnNames[1]]
print(classes.value_counts())

# pre-processing the data before classification

# convert the labels into binary values 
# where 0 = ham and 1 = spam
from sklearn.preprocessing import OneHotEncoder

binaryEncoder = OneHotEncoder()

DatasetY = classes.apply(binaryEncoder.fit_transform)
