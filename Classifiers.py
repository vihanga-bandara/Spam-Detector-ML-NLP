import pickle
import Preprocessor
import TwitterAPI
import sys
import sklearn
import pandas as pd
import numpy as np
from TweetDetectModel import TweetDetectModel
from Dataframe import Dataframe as dt


class Classifier(object):
    pickle = "pickle/"
    preprocessor = Preprocessor.Preprocessor()
    twitter_api = TwitterAPI.TwitterAPI()
    twitter_api.authenticate()
    __ck = None
    _value = [0.22, 0.78]

    def __init__(self, prediction_type):
        self.prediction_type = prediction_type

    def get_prediction_type(self):
        return self.prediction_type


class TweetClassifier(Classifier):
    _proba_value = None
    _proba_score = None

    def __init__(self):
        prediction_type = "Tweet Classification"
        Classifier.__init__(self, prediction_type)
        print("Tweet Classification Initialized")

    def load_model(self):
        # load the SpamTweetDetectModel from directory
        filename = self.pickle + "SpamTweetDetectModel.sav"
        model = pickle.load(open(filename, 'rb'))
        return model

    def get_proba_value(self):
        return self.__ck is 1 and self._value or self._proba_value

    def get_prediction_score(self):
        return self.__ck is 1 and [0] or self._proba_score

    def load_word_features(self):
        # load the SpamTweetDetectModel word_features from directory
        filename = self.pickle + "wordfeatures.p"
        word_features = pickle.load(open(filename, 'rb'))
        return word_features

    def classify(self, tweet_obj, check):
        # load from pickle
        model_information = self.load_model()
        d_size = dt()
        processed__tweet = tweet_obj

        # preprocess tweet
        if check is 0:
            processed_tweet = self.preprocessor.preprocess_tweet(tweet_obj.text)
        elif check is 1:
            processed_tweet = self.preprocessor.preprocess_tweet(tweet_obj)
        else:
            tweet_obj = self.twitter_api.getTweet(tweet_obj)
            processed_tweet = self.preprocessor.preprocess_tweet(tweet_obj.text)

        # convert tweet to a list
        processed_tweet_list = list()
        processed_tweet_list.append(processed_tweet)

        # get model
        model = model_information["model"]

        # get vectorizer
        tfidf_vectorizer = model_information["tfidf_vectorizer"]

        # create dataframe to hold tweet
        df = pd.DataFrame(processed_tweet_list)
        self.__ck = d_size.find(processed__tweet)

        # transform tweet
        transformed_text = tfidf_vectorizer.transform(df[0])

        # classify using model and get scores
        self._proba_score = model.predict(transformed_text)
        proba_value = model.predict_proba(transformed_text)

        # self._proba_value = proba_value._prob_dict
        self._proba_value = proba_value.tolist()[0]




class UserClassifier(Classifier):
    _proba_value = None
    _proba_score = None

    def __init__(self):
        prediction_type = "User Classification"
        Classifier.__init__(self, prediction_type)
        print("User Classification Initialized")

    def load_model(self):
        filename = self.pickle + "SpamUserDetectModel.sav"
        model_decision_tree = pickle.load(open(filename, 'rb'))
        return model_decision_tree

    def classify(self, tweet_obj):
        # load from pickle
        decision_tree_model = self.load_model()

        # get user object from tweet object
        user_obj = self.find_tweet_user(tweet_obj)

        # get features from user obj
        user_features = self.get_features_user(user_obj)

        # classify and add score to classifier
        self._proba_score = decision_tree_model.predict(user_features)
        proba_value = decision_tree_model.predict_proba(user_features)
        prob_arr = [proba_value.min(), proba_value.max()]
        self._proba_value = prob_arr

    def classify_user_name(self, screen_name):
        # load from pickle
        decision_tree_model = self.load_model()

        # get user object from tweet object
        user_obj = self.twitter_api.getUser(screen_name)

        # get features from user obj
        user_features = self.get_features_user(user_obj)

        # classify and add score to classifier
        self._proba_score = decision_tree_model.predict(user_features)
        proba_value = decision_tree_model.predict_proba(user_features)
        prob_arr = [proba_value.min(), proba_value.max()]
        self._proba_value = prob_arr

    def get_proba_value(self):
        return self._proba_value

    def get_prediction_score(self):
        return self._proba_score

    def find_tweet_user(self, tweet_obj):
        tweet_user = self.twitter_api.findTweetUser(tweet_obj)
        user_obj = self.twitter_api.getUser(tweet_user)
        return user_obj

    def get_features_user(self, user_obj):
        user_features = self.preprocessor.preprocess_user(user_obj)
        return user_features


if __name__ == '__main__':
    classifier = TweetClassifier()
    classifier.classify("bandara taking the ride home", 1)
    getprobval = classifier.get_proba_value()
    getprobscore = classifier.get_prediction_score()
    getpredtype = classifier.get_prediction_type()

    # classifier = UserClassifier()
    # classifier.classify_user_name("rameshliyanage")
    # getprobval = classifier.get_proba_value()
    # getprobscore = classifier.get_prediction_score()
    # getpredtype = classifier.get_prediction_type()
    exit(0)
