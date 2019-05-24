import sys
import sklearn
import pandas as pd
import numpy as np
import nltk
import matplotlib
import re
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

    # def train_model_test(self, dataset):
    #
    #     # X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.30,
    #     #                                                     stratify=dataset[1])
    #
    #     # split data into train and test sets
    #     X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.35)
    #
    #     # initialize TF-ID Vectorizer
    #     tfidf = TfidfVectorizer(
    #         analyzer='word', tokenizer=self.dummy_fun, preprocessor=self.dummy_fun,
    #         token_pattern=None, stop_words=None,
    #         ngram_range=(1, 1), use_idf=True)
    #
    #     features_set_train = tfidf.fit_transform(X_train)
    #
    #     features_set_test = tfidf.transform(X_test)
    #
    #     """Grid Search and Cross Validation to identify best parameters"""
    #
    #     """ Random Forest Classifier """
    #     # # create a dictionary of all values we want to test for n_estimators
    #     # rf_classifier = RandomForestClassifier()
    #     # params_rf = {"n_estimators": [100, 200],
    #     #              "max_features": ['auto', 'sqrt', 'log2'],
    #     #              "max_depth": [5, 6, 7, 8],
    #     #              "criterion": ['gini', 'entropy']
    #     #              }
    #     #
    #     # # use gridsearch to test all values for n_estimators
    #     # rf_gs = GridSearchCV(rf_classifier, params_rf, cv=5, n_jobs=2)
    #     # # fit model to training data
    #     # rf_gs.fit(features_set_train, y_train)
    #     #
    #     # # save best model
    #     # rf_best = rf_gs.best_estimator_
    #     # # check best n_estimators value
    #     # print(rf_gs.best_params_)
    #     #
    #     # """ Logistic Regression """
    #     #
    #     # # create a dictionary of all values we want to test for n_estimators
    #     # lr_classifier = LogisticRegression()
    #     # params_lr = {"C": [0.1, 1, 4, 10],
    #     #              "penalty": [str, 'l1', 'l2']
    #     #              }
    #     # param_grid = [
    #     #     {
    #     #         'C': [0.1, 1, 4, 10],
    #     #         'penalty': ['l1'], 'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
    #     #     },
    #     #     {'penalty': ['l2'], 'solver': ['newton-cg']},
    #     # ]
    #     #
    #     # # use gridsearch to test all values for n_estimators
    #     # lr_gs = GridSearchCV(lr_classifier, params_lr, cv=5, n_jobs=2)
    #     # # fit model to training data
    #     # lr_gs.fit(features_set_train, y_train)
    #     #
    #     # # save best model
    #     # lr_best = lr_gs.best_estimator_
    #     # # check best n_estimators value
    #     # print(lr_gs.best_params_)
    #
    #     """ END OF GRID SEARCH - best params have been used for model """
    #
    #     # create a new random forest classifier with best params from grid search
    #     rf_classifier = RandomForestClassifier(n_estimators=200, max_features="auto", criterion="gini", max_depth=7)
    #     rf_classifier.fit(features_set_train, y_train)
    #
    #     # predicting on the same trained set
    #     y_pred_train = rf_classifier.predict(features_set_train)
    #
    #     # predict on the test dataset
    #     y_pred_test = rf_classifier.predict(features_set_test)
    #
    #     # score of the test dataset
    #     y_score_test = rf_classifier.score(features_set_test, y_test)
    #
    #     # Output classifier results
    #     print("Training Accuracy: %.5f" % accuracy_score(y_train, y_pred_train))
    #     print("Test Accuracy: %.5f" % accuracy_score(y_test, y_pred_test))
    #
    #     # probability of prediction of test data using trained model
    #     y_pred_proba = rf_classifier.predict_proba(features_set_test)
    #     print(np.mean(y_pred_test == y_test))
    #
    #     train_set = pd.DataFrame()
    #     train_set[0] = X_train
    #     train_set[1] = y_train
    #     train_set[2] = y_pred_train
    #
    #     test_set = pd.DataFrame()
    #     test_set[0] = X_test
    #     test_set[1] = y_test
    #     test_set[2] = y_pred_test
    #
    #     # running cross validation score on testing data
    #     scores = cross_val_score(rf_classifier, features_set_test, y_test, scoring='accuracy', cv=5)
    #     accuracy = scores
    #     mean = scores.mean()
    #     std = scores.std()
    #
    #     print('Cross Validation Accuracy is {} | Mean is {} | Standard Deviation is {} on train data'.format(accuracy,
    #                                                                                                          mean, std))
    #
    #     return rf_classifier, tfidf, y_test, y_pred_test, y_pred_proba
    #
    #     # # create a new logistic regression model
    #     # log_reg = LogisticRegression()
    #     #
    #     # # fit the model to the training data
    #     # log_reg.fit(X_train, y_train)


    def load_dataset(self):
        # try to load the dataset
        try:
            df = pd.read_csv('data/SpamTweetsFinalDataset.csv', header=None)
            print(df.info())
            print(df.head())

            # sort data frame to avoid bias
            # df.sort_values(1, ascending=False)
            # df.sort_values(by=[1], inplace=True, ascending=False)
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

        # store as list of lists of words
        sentences_ted = []
        for sent_str in processed:
            tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
            sentences_ted.append(tokens)

        self.tweets_processed_col = sentences_ted

        return sentences_ted

    def label_encoding(self, classes_column):
        # convert the labels into binary values
        # where 0 = ham and 1 = spam
        label_encoder = LabelEncoder()
        self.tweets_class_col = label_encoder.fit_transform(classes_column)

    # dummy variable pass to avoid tokenizing and preprocessor since its already been done
    def dummy_fun(self, doc):
        return doc

    def train_model_realtime(self, dataset):

        # initialize TF-ID Vectorizer
        tfidf = TfidfVectorizer(
            analyzer='word', tokenizer=self.dummy_fun, preprocessor=self.dummy_fun,
            token_pattern=None, stop_words=None,
            ngram_range=(1, 1), use_idf=True)

        features_set_train = tfidf.fit_transform(dataset[0])

        self.display_scores(tfidf, features_set_train)

        # create a new random forest classifier with best params from grid search
        rf_classifier = RandomForestClassifier(n_estimators=200, max_features="auto", criterion="gini", max_depth=7)
        # fit train data to classifier
        rf_classifier.fit(features_set_train, dataset[1])

        return rf_classifier, tfidf

        # # create new a knn model

        # # create a new logistic regression model
        # log_reg = LogisticRegression()
        #
        # # fit the model to the training data
        # log_reg.fit(X_train, y_train)

    def train_model(self, dataset):

        # X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.30,
        #                                                     stratify=dataset[1])

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.35)

        # initialize TF-ID Vectorizer
        tfidf = TfidfVectorizer(
            analyzer='word', tokenizer=self.dummy_fun, preprocessor=self.dummy_fun,
            token_pattern=None, stop_words=None,
            ngram_range=(1, 1), use_idf=True)

        features_set_train = tfidf.fit_transform(X_train)

        features_set_test = tfidf.transform(X_test)

        """Grid Search and Cross Validation to identify best parameters"""

        """ Random Forest Classifier """
        # # create a dictionary of all values we want to test for n_estimators
        # rf_classifier = RandomForestClassifier()
        # params_rf = {"n_estimators": [100, 200],
        #              "max_features": ['auto', 'sqrt', 'log2'],
        #              "max_depth": [5, 6, 7, 8],
        #              "criterion": ['gini', 'entropy']
        #              }
        #
        # # use gridsearch to test all values for n_estimators
        # rf_gs = GridSearchCV(rf_classifier, params_rf, cv=5, n_jobs=2)
        # # fit model to training data
        # rf_gs.fit(features_set_train, y_train)
        #
        # # save best model
        # rf_best = rf_gs.best_estimator_
        # # check best n_estimators value
        # print(rf_gs.best_params_)
        #
        # """ Logistic Regression """
        #
        # # create a dictionary of all values we want to test for n_estimators
        # lr_classifier = LogisticRegression()
        # params_lr = {"C": [0.1, 1, 4, 10],
        #              "penalty": [str, 'l1', 'l2']
        #              }
        # param_grid = [
        #     {
        #         'C': [0.1, 1, 4, 10],
        #         'penalty': ['l1'], 'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
        #     },
        #     {'penalty': ['l2'], 'solver': ['newton-cg']},
        # ]
        #
        # # use gridsearch to test all values for n_estimators
        # lr_gs = GridSearchCV(lr_classifier, params_lr, cv=5, n_jobs=2)
        # # fit model to training data
        # lr_gs.fit(features_set_train, y_train)
        #
        # # save best model
        # lr_best = lr_gs.best_estimator_
        # # check best n_estimators value
        # print(lr_gs.best_params_)

        """ END OF GRID SEARCH - best params have been used for model """

        # create a new random forest classifier with best params from grid search
        rf_classifier = RandomForestClassifier(n_estimators=200, max_features="auto", criterion="entropy", max_depth=7)
        rf_classifier.fit(features_set_train, y_train)

        # create a new random forest classifier with best params from grid search
        lr_classifier = LogisticRegression()
        lr_classifier.fit(features_set_train, y_train)

        """ Testing for RandomForest Classifier """

        # predicting on the same trained set
        y_pred_train = rf_classifier.predict(features_set_train)

        # predict on the test dataset
        y_pred_test = rf_classifier.predict(features_set_test)

        # score of the test dataset
        y_score_test = rf_classifier.score(features_set_test, y_test)

        # Output classifier results
        print("Training Accuracy: %.5f" % accuracy_score(y_train, y_pred_train))
        print("Test Accuracy: %.5f" % accuracy_score(y_test, y_pred_test))

        # probability of prediction of test data using trained model
        y_pred_proba = rf_classifier.predict_proba(features_set_test)
        print(np.mean(y_pred_test == y_test))

        train_set = pd.DataFrame()
        train_set[0] = X_train
        train_set[1] = y_train
        train_set[2] = y_pred_train

        test_set = pd.DataFrame()
        test_set[0] = X_test
        test_set[1] = y_test
        test_set[2] = y_pred_test

        # running cross validation score on testing data
        scores = cross_val_score(rf_classifier, features_set_test, y_test, scoring='accuracy', cv=5)
        accuracy = scores
        mean = scores.mean()
        std = scores.std()

        print('Cross Validation Accuracy is {} | Mean is {} | Standard Deviation is {} on train data'.format(accuracy,
                                                                                                             mean, std))

        """ Testing END for RandomForest Classifier """

        """ Testing for Logistic Regression Classifier """

        # predicting on the same trained set
        y_pred_train = lr_classifier.predict(features_set_train)

        # predict on the test dataset
        y_pred_test = lr_classifier.predict(features_set_test)

        # score of the test dataset
        y_score_test = lr_classifier.score(features_set_test, y_test)

        # Output classifier results
        print("Training Accuracy: %.5f" % accuracy_score(y_train, y_pred_train))
        print("Test Accuracy: %.5f" % accuracy_score(y_test, y_pred_test))

        # probability of prediction of test data using trained model
        y_pred_proba = rf_classifier.predict_proba(features_set_test)
        print(np.mean(y_pred_test == y_test))

        train_set = pd.DataFrame()
        train_set[0] = X_train
        train_set[1] = y_train
        train_set[2] = y_pred_train

        test_set = pd.DataFrame()
        test_set[0] = X_test
        test_set[1] = y_test
        test_set[2] = y_pred_test

        # running cross validation score on testing data
        scores = cross_val_score(rf_classifier, features_set_test, y_test, scoring='accuracy', cv=5)
        accuracy = scores
        mean = scores.mean()
        std = scores.std()

        print('Cross Validation Accuracy is {} | Mean is {} | Standard Deviation is {} on train data'.format(accuracy,
                                                                                                             mean, std))
        """ Testing END for Logistic Regression Classifier """

        # create a dictionary of our models
        estimators = [("rf", rf_classifier), ("log_reg", lr_classifier)]
        # create our voting classifier, inputting our models
        ensemble = VotingClassifier(estimators, voting="soft")

        # fit model to training data
        ensemble.fit(features_set_train, y_train)
        # test our model on the test data
        ensemble_score = ensemble.score(features_set_test, y_test)
        print(ensemble_score)

        return rf_classifier, tfidf, y_test, y_pred_test, y_pred_proba

        # # create a new logistic regression model
        # log_reg = LogisticRegression()
        #
        # # fit the model to the training data
        # log_reg.fit(X_train, y_train)

    def generate_performance_reports(self, model, y_test, y_pred, y_pred_proba, tfidf_vectorizer):

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

    def save_model_pickle(self, model, tfidf_vectorizer):
        info = self.info

        # get current date time
        current_date_time = datetime.datetime.now()
        current_date_time_string = "{0} - {1}".format(current_date_time.strftime("%a, %b %d, %Y"),
                                                      current_date_time.strftime("%I:%M:%S %p"))

        # add model and model information to dictionary
        model_information = dict({
            'model': model,
            'tfidf_vectorizer': tfidf_vectorizer,
            'metadata': {
                'name': 'Tweet Spam Detection Model',
                'author': 'Vihanga Bandara',
                'date': current_date_time_string,
                'source_code_version': 'unreleased_{0}'.format(current_date_time.strftime("%I:%M:%S %p")),
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

        # save model information using pickle
        filename = 'pickle/SpamTweetDetectModel.sav'
        pickle.dump(model_information, open(filename, 'wb'))

        # load the SpamTweetDetectModel from directory
        filename = "pickle/SpamTweetDetectModel.sav"
        model = pickle.load(open(filename, 'rb'))

        print("Successfully saved model information to pickle")

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
                # train model
                model, tfidf_vectorizer = self.train_model_realtime(self.tweets_dataset)
                # save model to pickle
                self.save_model_pickle(model, tfidf_vectorizer)
            else:
                # split data and train model
                model, tfidf_vectorizer, y_test, y_pred_test, y_pred_proba = self.train_model(self.tweets_dataset)

                # generate performance reports
                self.generate_performance_reports(model, y_test, y_pred_test, y_pred_proba, tfidf_vectorizer)

                # save model to pickle
                self.save_model_pickle(model, tfidf_vectorizer)
            return True
        else:
            return False

    def classify(self, tweet):
        # load the SpamTweetDetectModel from directory
        filename = "pickle/SpamTweetDetectModel.sav"
        model_information = pickle.load(open(filename, 'rb'))

        # get model
        model = model_information["model"]

        # get vectorizer
        tfidf_vectorizer = model_information["tfidf_vectorizer"]

        # preprocess tweet
        # preprocessor = Preprocessor()

        tweet_df = pd.DataFrame()
        tweet_df[0] = [tweet]

        tweet_df[0] = self.preprocessing_tweets(tweet_df[0])

        processed_tweet = tweet_df[0]

        # transform tweet
        transformed_text = tfidf_vectorizer.transform(processed_tweet)

        # get proba score and value after predicting
        proba_score = model.predict(transformed_text)
        proba_value = model.predict_proba(transformed_text)

        # convert proba value to list
        proba_values = proba_value.tolist()[0]

        return proba_score, proba_values, tweet

    def display_scores(self, vectorizer, tfidf_result):
        scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tfidf_result.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        spam_tokens = []
        for item in sorted_scores:
            # print("{0:50} Score: {1}".format(item[0], item[1]))
            if type(item[0]) == str and item[1] > 1.8:
                spam_tokens.append(item[0])
                # print('adding spam token - {0}'.format(item[0]))


if __name__ == '__main__':
    train = TweetDetectModel()

    # #run model manually
    # train.main(1)
    # train.main(0)
    train.main(1)

    # print(train.classify('Click to check your daily and become rich'))
    # print(train.classify('Here is a small gift for you #gifts'))
    # print(train.classify('Best investment from us, retweet to win'))
    # print(train.classify('Obtain complimentary coin, check it out now'))

    # processed_tweet = ['this life is horrible right now','check me out now']
    # df = pd.DataFrame()
    # df[0] = processed_tweet
    # print(df)
