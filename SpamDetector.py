import pandas as pd
import numpy as np
import nltk
import pickle
import TwitterAPI
import Preprocessor
import SpamFuzzyController
import DriftDetector
import re
from nltk.tokenize import word_tokenize


class SpamDetector:
    def load_models(self):
        # load the SpamTweetDetectModel word_features from directory
        pickle = "pickle/"
        filename = pickle + "wordfeatures.p"
        word_features = pickle.load(open(filename, 'rb'))

        # load the SpamTweetDetectModel from directory
        filename = pickle + "SpamTweetDetectModel.sav"
        nltk_ensemble = pickle.load(open(filename, 'rb'))

    def main(self, tweet_obj, tweet_id, tweet_only):


tweet = preprocess(tweetObj.text)

features_test = find_features(tweet)
prediction_test = nltk_ensemble.classify(features_test)
tweet_model_score = nltk_ensemble.prob_classify(features_test)
labels = nltk_ensemble.labels()

print("Spam User Model Prediction = {0}".format(prediction_test))

# retrieve spam user account detail by using tweet handle
# for now using a known spam account to retrieve the data

# since api is already initialised and authenticated retrieve user object details
twitterUser = twitter.findTweetUser(tweetObj)
userObj = twitter.getUser(twitterUser)
print(userObj)


def convertUserDetails(userObj):
    # create a dataframe
    data = [
        [userObj.screen_name, userObj.name, userObj.description, userObj.status, userObj.listed_count, userObj.verified,
         userObj.followers_count, userObj.friends_count, userObj.statuses_count]]
    data = pd.DataFrame(data, columns=['screen_name', 'name', 'description', 'status', 'listed_count', 'verified',
                                       'followers_count', 'friends_count', 'statuses_count'])

    # load the SpamUserDetectModel bag-of-words-bot from directory
    filename = "bagofwords.p"
    bag_of_words_bot = pickle.load(open(filename, 'rb'))

    # Feature Engineering (some more relationships to be added)

    # check the screen name for words in the BoW
    data['screen_name_binary'] = data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)

    # check the name for words in the BoW
    data['name_binary'] = data.name.str.contains(bag_of_words_bot, case=False, na=False)

    # check the description for words in the BoW
    data['description_binary'] = data.description.str.contains(bag_of_words_bot, case=False, na=False)

    # check the sstatus for words in the BoW
    data['status_binary'] = data.status.str.contains(bag_of_words_bot, case=False, na=False)

    # check the number of public lists that the user is a part of
    data['listed_count_binary'] = (data.listed_count > 20000) == False

    # check whether account is verified or not

    # Finalizing the feature set without independent variable 'bot'
    features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified',
                'followers_count',
                'friends_count', 'statuses_count', 'listed_count_binary']

    # customize dataset according to feature
    features_set = data[features]
    return features_set


# load the SpamUserDetectModel from directory
filename = "SpamUserDetectModel.sav"
DecisionTreeClf = pickle.load(open(filename, 'rb'))

# get the feature set from user obj
features = convertUserDetails(userObj)

# predict whether its a spam user or not
SpamUserModelPrediction = DecisionTreeClf.predict(features)
user_model_score = DecisionTreeClf.predict_proba(features)
print("Spam User Model Prediction = {0}".format(SpamUserModelPrediction))

preprocessor = Preprocessor.Preprocessing()
tweet = preprocessor.preprocess_tweet(tweetObj.text)
tweet_tokens = word_tokenize(tweet)
# send two model output proba through fuzzy logic controller to determine final output
fuzzy_system = SpamFuzzyController.SpamFuzzyController()
fuzzy_system.fuzzy_initialize()
spam_score_fuzzy = fuzzy_system.fuzzy_predict(10, 10)
print('Fuzzy Controller predicts {} of the user and twitter connection for spam activity'.format(spam_score_fuzzy))

if spam_score_fuzzy > 50:
    print("Show the Tweet along with the option of user manually reporting")
else:
    print("Sending the tweet to check for drift using drift detector")
    # if tweet is not spam as defined by fuzzy logic
    drift_detector = DriftDetector.DriftDetector()
    drift_detector.predict(tweet_tokens)
    print('done')
