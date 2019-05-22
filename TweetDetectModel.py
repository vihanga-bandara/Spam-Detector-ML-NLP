import sys
import sklearn
import pandas as pd
import numpy as np
import nltk
import matplotlib
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import datetime
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


class TweetDetectModel:
    tweets_dataset = None
    tweets_class_col = None
    tweets_col = None
    tweets_processed_col = None
    info = ['92.6', '0.906', '0.946', '0.925', ['106|6|11|107'], '0.96']

    def get_package_versions(self):
        package_version = dict()
        package_version["Python"] = 'Python: {}', format(sys.version)
        package_version["Python"] = 'NLTK: {}', format(nltk.__version__)
        package_version["Python"] = 'Scikit-learn: {}', format(sklearn.__version__)
        package_version["Python"] = 'Pandas: {}', format(pd.__version__)
        package_version["Python"] = 'Numpy: {}', format(np.__version__)
        package_version["Python"] = 'MatPlotLib: {}', format(matplotlib.__version__)
        return package_version

    def load_dataset(self):
        # try to load the dataset
        try:
            df = pd.read_csv('dataset/SpamTweetsFinalDataset.csv', header=None)
            print(df.info())
            print(df.head())

            # check the class balance ratio / distribution
            df_spam = df.loc[df[1] == 'spam']
            df_ham = df.loc[df[1] == 'ham']
            ratio = int(df.count()[0] / df_spam.count()[0])
            print("ratio of spam to to total data is {0}".format(ratio))

            # separate label column to add numerical label
            column_names = list(df.head(0))
            classes_column = df[column_names[1]].str.strip()
            tweets_column = df[column_names[0]].str.strip()

            # store dataset in object
            self.tweets_dataset = df
            # store class column in object
            self.tweets_class_col = classes_column
            # store tweet column in object
            self.tweets_col = tweets_column

            return True

        except Exception as e:
            # if loading dataset does not work return false
            print("Error loading the dataset")

            return False

    def preprocessing_tweets(self, unprocessed_tweets):

        # pre-processing the tweets before classification
        # using regex to identify different combinations in the tweet

        # replacing email addresses with 'emailaddr'
        processed = unprocessed_tweets.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

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

        stop_words = set(stopwords.words('english'))

        processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

        # using a Porter stemmer to remove word stems
        ps = nltk.PorterStemmer()
        processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

        self.tweets_processed_col = processed
        return processed

    def label_encoding(self, classes_column):
        # convert the labels into binary values
        # where 0 = ham and 1 = spam
        label_encoder = LabelEncoder()
        self.tweets_class_col = label_encoder.fit_transform(classes_column)

    def train_model(self, dataset):
        # split the data to train and test
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.25)

        # create pipeline that will consecutively carry out the training process
        pipeline = Pipeline(
            [('vectorizer', CountVectorizer()),
             ('tfidf', TfidfTransformer()),
             ('classifier', RandomForestClassifier())])

        train_score = pipeline.fit(X_train, y_train)
        print(train_score)

        # predict test data using trained model
        y_pred = pipeline.predict(X_test)
        # probability of prediction of test data using trained model
        y_pred_proba = pipeline.predict_proba(X_test)
        print(np.mean(y_pred == y_test))

        return pipeline, y_test, y_pred, y_pred_proba

    def generate_performance_reports(self, pipeline, y_test, y_pred, y_pred_proba):

        # get confusion matrix
        # from sklearn.metrics import confusion_matrix
        # confusion_matrix = pd.DataFrame(
        #     confusion_matrix(y_test, y_pred),
        #     index=[['actual', 'actual '], ['ham', 'spam']],
        #     columns=[['predicted', 'predicted'], ['ham', 'spam']])

        # get graphs
        cm = confusion_matrix(y_test, y_pred)
        ax = plt.subplot()
        svm = sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Greens')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['ham', 'spam'])
        ax.yaxis.set_ticklabels(['ham', 'spam'])
        figure = svm.get_figure()
        figure.savefig('images/confusion_matrix.png', dpi=400)
        plt.close()

        # keep probabilities for the positive outcome only
        probs = y_pred_proba[:, 1]

        # calculate AUC Score
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

    def save_model_pickle(self, pipeline):
        info = self.info

        # get current date time
        current_date_time = datetime.datetime.now()
        current_date_time_string = "{0} - {1}".format(current_date_time.strftime("%a, %b %d, %Y"),
                                                      current_date_time.strftime("%I:%M:%S %p"))

        # add model and model information to dictionary
        model_information = dict({
            'model': pipeline,
            'metadata': {
                'name': 'Tweet Spam Detection Model Pipeline',
                'author': 'Vihanga Bandara',
                'date': current_date_time_string,
                'source_code_version': 'unreleased_1',
                'metrics': {
                    'tweet_model_accuracy': info[0],
                    'recall': info[1],
                    'precision': info[2],
                    'f_measure': info[3],
                    'confusion_matrix_scores': info[4],
                    'auc_score': info[5]
                }
            }
        })

        # save model using pickle
        filename = 'SpamTweetDetectModel.sav'
        pickle.dump(model_information, open(filename, 'wb'))
        return True

    def main(self):
        # get package version details
        package_version = self.__get_package_versions()

        # load dataset
        if self.load_dataset():
            # encode classes using LabelEncoder
            self.label_encoding(self.tweets_class_col)
            unprocessed_tweets = self.tweets_col
            self.preprocessing_tweets(unprocessed_tweets)

            # finalising dataset with labelled classes and processed tweets
            self.tweets_dataset[1] = self.tweets_class_col
            self.tweets_dataset[0] = self.tweets_processed_col

            # split data and train model using pipeline
            pipeline, y_test, y_pred, y_pred_proba = self.train_model(self.tweets_dataset)

            # generate performance reports
            self.generate_performance_reports(pipeline, y_test, y_pred, y_pred_proba)

            # save model to pickle
            self.save_model_pickle(pipeline)
            from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


        else:
            return False
