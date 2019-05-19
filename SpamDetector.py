from SpamFuzzyController import SpamFuzzyController
from Classifiers import TweetClassifier, UserClassifier
from DriftDetector import DriftDetector
from TwitterAPI import TwitterAPI
import pickle
from timeit import default_timer as timer


class SpamDetector:
    information_array = dict()
    """ check = 0 - Tweet Object is sent
        check = 1 - Tweet is sent
        check = 2 - Tweet ID is sent """
    check = None

    def __init__(self):
        print("Spam Detector Framework Initialized")

    def main(self, tweet_obj, functionality_check):
        """ checks
        if check = 1 >>> full detection with custom created user (listened tweet object is sent)
                                                OR
                         full detection with normal user (normal tweet object is sent)

        if check = 2 >>> tweet detection only (only tweet detection model will be used, tweet is sent)"""

        # check for availability of variable
        if 'tweet_obj' in locals() and tweet_obj is not None and functionality_check is 1:
            """ Full detection with custom created user/ normal user
                Functionality - Tweet Classification, User Classification, Drift Detection"""

            """ Tweet Classification """

            """check = 0 means full object , check = 1 means tweet text, check = 2 means tweet id"""

            self.check = 0

            # initialize tweet classification
            tweet_classifier = TweetClassifier()

            # classify tweet and get prediction
            tweet_classifier.classify(tweet_obj, self.check)
            tweet_prediction_score = tweet_classifier.get_prediction_score()
            tweet_prediction_proba = tweet_classifier.get_proba_value()

            print("Tweet Spam Prediction = {0}".format(tweet_prediction_score))
            print("Tweet Spam Probabilities = spam({0}) ham({1})".format(tweet_prediction_proba[1],
                                                                         tweet_prediction_proba[0]))

            """ User Classification """

            # initialize user classification
            user_classifier = UserClassifier()

            # classify user and get prediction
            user_classifier.classify(tweet_obj)
            user_prediction_score = user_classifier.get_prediction_score()
            user_prediction_proba = user_classifier.get_proba_value()

            print("User Spam Prediction = {0}".format(user_prediction_score))
            print("User Spam Probabilities = spam({0}) ham({1})".format(user_prediction_proba[1],
                                                                        user_prediction_proba[0]))

            """ Fuzzy Logic """

            # send two model output proba through fuzzy logic controller to determine final output
            fuzzy_system = SpamFuzzyController()
            fuzzy_system.fuzzy_initialize()
            spam_score_fuzzy = fuzzy_system.fuzzy_predict(tweet_prediction_proba[1] * 100,
                                                          user_prediction_proba[1] * 100)
            print('Fuzzy Controller predicts {} of the user and twitter connection for spam activity'.format(
                spam_score_fuzzy))

            """ Basic flow ends here, displays spam predictions from both models 
                        and then the combined spam value using fuzzy logic       """

            """ Drift Detection """

            # drift_report = self.drift_detection(tweet_prediction_score, tweet_obj)

            self.information_array = self.info_prediction(user_prediction_score.min(), user_prediction_proba[1],
                                                          tweet_prediction_score, tweet_prediction_proba[1],
                                                          spam_score_fuzzy, tweet_obj, None)

        if 'tweet_obj' in locals() and tweet_obj is not None and functionality_check is 2:
            """ Tweet Classification """
            self.check = 1
            # initialize tweet classification
            tweet_classifier = TweetClassifier()

            # classify tweet and get prediction
            tweet_classifier.classify(tweet_obj, self.check)
            tweet_prediction_score = tweet_classifier.get_prediction_score()
            tweet_prediction_proba = tweet_classifier.get_proba_value()

            print("Tweet Spam Prediction = {0}".format(tweet_prediction_score))
            print("Tweet Spam Probabilities = spam({0}) ham({1})".format(tweet_prediction_proba[1],
                                                                         tweet_prediction_proba[0]))
            drift_report = self.drift_detection(tweet_prediction_score, tweet_obj)

            self.information_array = self.info_prediction(None, tweet_prediction_score, tweet_prediction_proba[1],
                                                          None, None, tweet_obj, drift_report)

        # if 'tweet_id' in locals() and tweet_id is not None:
        #
        #     """ Tweet Classification """
        #     # initialize tweet classification
        #     tweet_classifier = TweetClassifier()
        #
        #     # classify tweet and get prediction
        #     tweet_classifier.classify(tweet_obj, 2)
        #     tweet_prediction_score = tweet_classifier.get_prediction_score()
        #     tweet_prediction_proba = tweet_classifier.get_proba_value()
        #
        #     print("Tweet Spam Prediction = {0}".format(tweet_prediction_score))
        #     print("Tweet Spam Probabilities = {0}".format(tweet_prediction_proba))
        #
        #     """ User Classification """
        #
        #     # initialize user classification
        #     user_classifier = UserClassifier()
        #
        #     # classify user and get prediction
        #     user_classifier.classify(tweet_obj)
        #     user_prediction_score = user_classifier.get_prediction_score()
        #     user_prediction_proba = user_classifier.get_proba_value()
        #
        #     print("User Spam Prediction = {0}".format(user_prediction_score))
        #     print("User Spam Probabilities = {0}".format(user_prediction_proba))
        #
        #     """ Fuzzy Logic """
        #
        #     # send two model output proba through fuzzy logic controller to determine final output
        #     fuzzy_system = SpamFuzzyController()
        #     fuzzy_system.fuzzy_initialize()
        #     spam_score_fuzzy = fuzzy_system.fuzzy_predict(tweet_prediction_proba, user_prediction_proba)
        #     print('Fuzzy Controller predicts {} of the user and twitter connection for spam activity'.format(
        #         spam_score_fuzzy))
        #
        #     """ Basic flow ends here, displays spam predictions from both models
        #                 and then the combined spam value using fuzzy logic       """
        #
        #     """ Drift Detection """
        #
        #     if tweet_prediction_score is not 1:
        #         print("Checking Tweet for Drift Possibility")
        #         # drift_detector = DriftDetector()
        #         # drift_detector.predict(tweet_obj)
        #         # print('done')
        #
        #     self.information_array = self.get_info_prediction(user_prediction_score,
        #                                                       tweet_prediction_score, spam_score_fuzzy)

    def info_prediction(self, user_score, user_percentage, tweet_score, tweet_percentage, fuzzy_score, tweet_obj,
                        drift_report):
        information_array = dict()
        if user_score is not None and user_score == 1:
            information_array['user_prediction'] = "spam"
        elif user_score is not None and user_score == 0:
            information_array['user_prediction'] = "ham"

        if tweet_score is not None and tweet_score == 1:
            information_array['tweet_prediction'] = "spam"
        elif tweet_score is not None and tweet_score == 0:
            information_array['tweet_prediction'] = "ham"

        if user_percentage is not None:
            information_array['user_percentage'] = int(round(user_percentage))

        if tweet_percentage is not None:
            information_array['tweet_percentage'] = int(round(tweet_percentage))

        if fuzzy_score is not None:
            information_array['fuzzy_score'] = fuzzy_score

        if drift_report is not None and len(drift_report) > 0:
            information_array["spam_status"] = drift_report["spam_status"]
            information_array["spam_score"] = drift_report["spam_score"]

        if self.check == 0:
            information_array["tweet"] = tweet_obj.text
            information_array["user"] = tweet_obj.user.screen_name
            information_array["user_image"] = tweet_obj.user.profile_image_url
        elif self.check == 1:
            # preprocessor = Preprocessor()
            # processed_tweet = preprocessor.preprocess_tweet(tweet_obj)
            information_array["tweet"] = tweet_obj

        return information_array

    def get_prediction_report(self):
        return self.information_array

    def drift_detection(self, tweet_prediction_score, tweet_obj):
        drift_report = dict()
        if tweet_prediction_score != 1:
            print("Checking Tweet for Drift Possibility")
            drift_detector = DriftDetector()
            start = timer()
            drift_report = drift_detector.predict(tweet_obj, self.check)
            # calculate elapsed time
            end = timer()
            print("Drift Algorithm executed in {0} seconds".format(end - start))

        # if drift report is positive then use that prediction score for output
        if drift_report["spam_status"] == "Positive":
            tweet_prediction_score = drift_report["spam_score"]
        return drift_report


if __name__ == '__main__':
    twitter_api = TwitterAPI()
    twitter_api.authenticate()
    spamdet = SpamDetector()
    tweetObj = twitter_api.getTweet("1125183305265991681")
    spamdet.main(tweetObj, None, None)
    classification_report = spamdet.get_prediction_report()
    exit(0)
