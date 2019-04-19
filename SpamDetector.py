from SpamFuzzyController import SpamFuzzyController
from Classifiers import TweetClassifier, UserClassifier
from DriftDetector import DriftDetector
from TwitterAPI import TwitterAPI
import pickle


class SpamDetector:
    information_array = dict()
    """ check = 0 - Tweet Object is sent
        check = 1 - Tweet is sent
        check = 2 - Tweet ID is sent """
    check = None

    def __init__(self):
        print("Spam Detector Framework Initialized")

    def main(self, tweet_obj, tweet_only, tweet_id=None):
        # check for availability of variable
        if 'tweet_obj' in locals() and tweet_obj is not None:

            """ Tweet Classification """
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
            print("User Spam Probabilities = spam({0}) ham({1})".format(user_prediction_proba[0],
                                                                        user_prediction_proba[1]))

            """ Fuzzy Logic """

            # send two model output proba through fuzzy logic controller to determine final output
            fuzzy_system = SpamFuzzyController()
            fuzzy_system.fuzzy_initialize()
            spam_score_fuzzy = fuzzy_system.fuzzy_predict(tweet_prediction_proba[1] * 100,
                                                          user_prediction_proba[0] * 100)
            print('Fuzzy Controller predicts {} of the user and twitter connection for spam activity'.format(
                spam_score_fuzzy))

            """ Basic flow ends here, displays spam predictions from both models 
                        and then the combined spam value using fuzzy logic       """

            """ Drift Detection """

            # if tweet_prediction_score is not 1:
            #     print("Checking Tweet for Drift Possibility")
            #     drift_detector = DriftDetector()
            #     drift_detector.predict(tweet_obj)
            #     print('done')

            self.information_array = self.info_prediction(user_prediction_score.min(), tweet_prediction_score,
                                                          spam_score_fuzzy, tweet_obj)

        if 'tweet_only' in locals() and tweet_only is not None:
            """ Tweet Classification """
            self.check = 1
            # initialize tweet classification
            tweet_classifier = TweetClassifier()

            # classify tweet and get prediction
            tweet_classifier.classify(tweet_only, self.check)
            tweet_prediction_score = tweet_classifier.get_prediction_score()
            tweet_prediction_proba = tweet_classifier.get_proba_value()

            print("Tweet Spam Prediction = {0}".format(tweet_prediction_score))
            print("Tweet Spam Probabilities = spam({0}) ham({1})".format(tweet_prediction_proba[1],
                                                                         tweet_prediction_proba[0]))

            if tweet_prediction_score is not 1:
                print("Checking Tweet for Drift Possibility")
                drift_detector = DriftDetector()
                drift_detector.predict(tweet_only, self.check)
                print('done')

            self.information_array = self.info_prediction(tweet_prediction_score, tweet_only)

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

    def info_prediction(self, user_score, tweet_score, fuzzy_score, tweet_obj):
        information_array = dict()
        if user_score is 1 and user_score is not None:
            information_array['user_prediction'] = "spam"
        else:
            information_array['user_prediction'] = "ham"

        if tweet_score is 1 and tweet_score is not None:
            information_array['tweet_prediction'] = "spam"
        else:
            information_array['tweet_prediction'] = "ham"
        if fuzzy_score is not None:
            information_array['fuzzy_score'] = fuzzy_score

        if self.check is 0:
            information_array["tweet"] = tweet_obj.text
            information_array["user"] = tweet_obj.user.screen_name
            information_array["user_image"] = tweet_obj.user.profile_image_url
        else:
            information_array["tweet"] = tweet_obj

        return information_array

    def get_prediction_report(self):
        return self.information_array


if __name__ == '__main__':
    SpamDetector.main()
    exit(0)
