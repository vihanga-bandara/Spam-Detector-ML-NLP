import sys
import sklearn
import pandas as pd
import numpy as np
import nltk
import matplotlib
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
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
from Preprocessor import Preprocessor


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

            # sort data frame to avoid bias
            # df.sort_values(1, ascending=False)
            df.sort_values(by=[1], inplace=True, ascending=False)
            # check the class balance ratio / distribution
            df_spam = df.loc[df[1] == 'spam']
            df_ham = df.loc[df[1] == 'ham']

            ratio = float("{0:.2f}".format(df_spam.count()[0] / df.count()[0]))
            print("Ratio of Spam Tweets to Total Tweets is {0}".format(ratio))

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
            print("Error loading the dataset", str(e))

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
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.40)

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

        # running cross validation score on training data
        scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=10)
        accuracy = scores
        mean = scores.mean()
        std = scores.std()

        print('Accuracy is {} | Mean is {} | Standard Deviation is {} on train data'.format(accuracy, mean, std))

        # running cross validation score on all data
        scores_full = cross_val_score(pipeline, dataset[0], dataset[1], scoring='accuracy', cv=10)
        accuracy_full = scores_full.mean() * 100
        mean_full = scores_full.mean()
        std_full = scores_full.std()

        print('Accuracy is {} | Mean is {} | Standard Deviation is {} on all data'.format(accuracy_full, mean_full,
                                                                                          std_full))

        return pipeline, y_test, y_pred, y_pred_proba

    def train_model_realtime(self, dataset):

        # create pipeline that will consecutively carry out the training process
        pipeline = Pipeline(
            [('vectorizer', CountVectorizer()),
             ('tfidf', TfidfTransformer()),
             ('classifier', RandomForestClassifier())])

        pipeline.fit(dataset[0], dataset[1])

        # running cross validation score on full data
        scores_full = cross_val_score(pipeline, dataset[0], dataset[1], scoring='accuracy', cv=10)
        accuracy_full = scores_full.mean() * 100
        mean_full = scores_full.mean()
        std_full = scores_full.std()

        print('Accuracy is {} | Mean is {} | Standard Deviation is {} on all data'.format(accuracy_full, mean_full,
                                                                                          std_full))
        return pipeline

    def train_model_realtime_test(self, dataset):

        # create pipeline that will consecutively carry out the training process
        # check for train test split
        # dummy variable pass to avoid tokenizing and preprocessor since its already been done
        # X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.30,
        #                                                     stratify=dataset[1])

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.30)

        def dummy_fun(doc):
            return doc

        # initialize TF-ID Vectorizer
        tfidf = TfidfVectorizer(
            analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun,
            token_pattern=None, stop_words=None,
            ngram_range=(1, 1), use_idf=True)

        features_set_train = tfidf.fit_transform(X_train)

        features_set_test = tfidf.transform(X_test)

        # create a new random forest classifier
        rf_classifier = RandomForestClassifier()
        # rf_classifier.fit(features_set_train, y_train)

        # create a dictionary of all values we want to test for n_estimators
        params_rf = {"n_estimators": [200, 500],
                     "max_features": ['auto', 'sqrt', 'log2'],
                     "max_depth": [4, 5, 6, 7, 8],
                     "criterion": ['gini', 'entropy']
                     }

        # use gridsearch to test all values for n_estimators
        rf_gs = GridSearchCV(rf_classifier, params_rf, cv=5)
        # fit model to training data
        rf_gs.fit(features_set_train, y_train)

        # save best model
        rf_best = rf_gs.best_estimator_
        # check best n_estimators value
        print(rf_gs.best_params_)

        # predicting on the same trained set
        y_pred_train = rf_classifier.predict(features_set_train)

        # predict on the test dataset
        y_pred_test = rf_classifier.predict(features_set_test)

        # score of the test dataset
        y_score_test = rf_classifier.score(features_set_test, y_test)
        print(y_score_test)

        # Output classifier results
        print("Training Accuracy: %.5f" % accuracy_score(y_train, y_pred_train))
        print("Test Accuracy: %.5f" % accuracy_score(y_test, y_pred_test))

        data = ['webaddr do you want to go out with me']
        df = pd.DataFrame(data)

        check_test = tfidf.transform(df[0])

        check_model_predict = rf_classifier.predict(check_test)

        return tfidf

        # # create new a knn model
        # knn = KNeighborsClassifier()
        #
        # # create a dictionary of all values we want to test for n_neighbors
        # params_knn = {"n_neighbors": np.arange(1, 25)}
        # # use gridsearch to test all values for n_neighbors
        # knn_gs = GridSearchCV(knn, params_knn, cv=5)
        # # fit model to training data
        # knn_gs.fit(X_train, y_train)
        #
        # # save best model
        # knn_best = knn_gs.best_estimator_
        # # check best n_neigbors value
        # print(knn_gs.best_params_)

        # # create a new logistic regression model
        # log_reg = LogisticRegression()
        #
        # # fit the model to the training data
        # log_reg.fit(X_train, y_train)

        # # running cross validation score on full data
        # scores_full = cross_val_score(pipeline, dataset[0], dataset[1], scoring='accuracy', cv=10)
        # accuracy_full = scores_full.mean() * 100
        # mean_full = scores_full.mean()
        # std_full = scores_full.std()
        #
        # print('Accuracy is {} | Mean is {} | Standard Deviation is {} on all data'.format(accuracy_full, mean_full,
        #                                                                                   std_full))
        # return pipeline

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
        filename = 'pickle/SpamTweetDetectModel.sav'
        pickle.dump(model_information, open(filename, 'wb'))
        return True

    def main(self, check):
        # get package version details
        package_version = self.get_package_versions()

        # load dataset
        if self.load_dataset():
            # encode classes using LabelEncoder
            self.label_encoding(self.tweets_class_col)
            unprocessed_tweets = self.tweets_col
            self.preprocessing_tweets(unprocessed_tweets)

            # finalising dataset with labelled classes and processed tweets
            self.tweets_dataset[1] = self.tweets_class_col
            self.tweets_dataset[0] = self.tweets_processed_col

            """check = 0 means realtime model check = 1 is for performance testing"""

            if check == 0:
                # train model using pipeline
                pipeline = self.train_model_realtime_test(self.tweets_dataset)
                # save model to pickle
                self.save_model_pickle(pipeline)
            else:
                # split data and train model using pipeline
                pipeline, y_test, y_pred, y_pred_proba = self.train_model(self.tweets_dataset)
                # generate performance reports
                self.generate_performance_reports(pipeline, y_test, y_pred, y_pred_proba)

        else:
            return False

    def classify(self, tweet):
        # load the SpamTweetDetectModel from directory
        filename = "pickle/SpamTweetDetectModel.sav"
        pipeline_ensemble = pickle.load(open(filename, 'rb'))

        # get model
        pipeline_model = pipeline_ensemble["model"]

        # preprocess tweet
        preprocessor = Preprocessor()
        processed_tweet = list()
        processed_tweet.append(preprocessor.preprocess_tweet(tweet))

        # get proba score and value after predicting
        proba_score = pipeline_model.predict(processed_tweet)
        proba_value = pipeline_model.predict_proba(processed_tweet)

        # convert proba value to list
        proba_values = proba_value.tolist()[0]

        return proba_score, proba_values, tweet


if __name__ == '__main__':
    train = TweetDetectModel()
    train.main(0)
    res = train.classify('Obtain complimentary coin, check it out now')
    print(res)
