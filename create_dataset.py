import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

# load the dataset
df = pd.read_csv('dataset/twitter-spam/train.csv')

new_df = df[['Tweet', 'Type']].copy()

from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

new_df['Type'] = labelEncoder.fit_transform(new_df['Type'])

new_df.info()

spam_tweets = new_df[new_df.Type == 1]
non_spam_tweets = new_df[new_df.Type == 0]
# store the twitter data
tweets = new_df['Tweet'].str.strip()

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

df['Tweet'] = processed

X = df['Tweet']
y = df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

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
np.mean(y_pred == y_test)
score = pipeline.predict(input)
print(score)
print(classification_report(y_test, y_pred))

spam_tweets = spam_tweets['Tweet']
non_spam_tweets = non_spam_tweets['Tweet']

tweets = list()
tweets = non_spam_tweets
my_list = list()
for tweet in spam_tweets:
    print(tweet)
    my_list.append(tweet)
import csv


def write_to_csv(list):
    with open('dataset/new_dataset_non_spam.csv', 'w') as csvfile:
        for domain in list:
            csvfile.write(domain + '\n')


write_to_csv(my_list)
