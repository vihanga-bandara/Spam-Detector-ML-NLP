from SpamFuzzyController import SpamFuzzyController
from Classifiers import TweetClassifier, UserClassifier


class SpamDetector:
    information_array = dict()
    """ check = 0 - Tweet Object is sent
        check = 1 - Tweet is sent
        check = 2 - Tweet ID is sent """
    check = None

    def __init__(self):
        print("Spam Detector Framework Initialized")

    def main(self, tweet_obj, tweet_id, tweet_only=None):
        # check for availability of variable
        if 'tweet_obj' in locals() and tweet_obj is not None:

            """ Tweet Classification """
            # initialize tweet classification
            tweet_classifier = TweetClassifier()

            # classify tweet and get prediction
            tweet_classifier.classify(tweet_obj, 0)
            tweet_prediction_score = tweet_classifier.get_prediction_score()
            tweet_prediction_proba = tweet_classifier.get_proba_value()

            print("Tweet Spam Prediction = {0}".format(tweet_prediction_score))
            print("Tweet Spam Probabilities = {0}".format(tweet_prediction_proba))

            """ User Classification """

            # initialize user classification
            user_classifier = UserClassifier()

            # classify user and get prediction
            user_classifier.classify(tweet_obj)
            user_prediction_score = user_classifier.get_prediction_score()
            user_prediction_proba = user_classifier.get_proba_value()

            print("User Spam Prediction = {0}".format(user_prediction_score))
            print("User Spam Probabilities = {0}".format(user_prediction_proba))

            """ Fuzzy Logic """

            # send two model output proba through fuzzy logic controller to determine final output
            fuzzy_system = SpamFuzzyController()
            fuzzy_system.fuzzy_initialize()
            spam_score_fuzzy = fuzzy_system.fuzzy_predict(tweet_prediction_proba, user_prediction_proba)
            print('Fuzzy Controller predicts {} of the user and twitter connection for spam activity'.format(
                spam_score_fuzzy))

            """ Basic flow ends here, displays spam predictions from both models 
                        and then the combined spam value using fuzzy logic       """

            """ Drift Detection """

            if tweet_prediction_score is not 1:
                print("Checking Tweet for Drift Possibility")
                # drift_detector = DriftDetector()
                # drift_detector.predict(tweet_obj)
                # print('done')

            self.information_array = self.get_info_prediction(user_prediction_score,
                                                              tweet_prediction_score, spam_score_fuzzy)

        if 'tweet_only' in locals() and tweet_only is not None:

            """ Tweet Classification """
            # initialize tweet classification
            tweet_classifier = TweetClassifier()

            # classify tweet and get prediction
            tweet_classifier.classify(tweet_obj, 1)
            tweet_prediction_score = tweet_classifier.get_prediction_score()
            tweet_prediction_proba = tweet_classifier.get_proba_value()

            print("Tweet Spam Prediction = {0}".format(tweet_prediction_score))
            print("Tweet Spam Probabilities = {0}".format(tweet_prediction_proba))

            self.information_array['tweet_prediction'] = tweet_prediction_score

        # if 'tweet_obj' in locals() and tweet_obj is not None:
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

    def info_prediction(self, user_score, tweet_score, fuzzy_score):
        information_array = dict()
        if user_score is 1:
            information_array['user_prediction'] = "spam"
        else:
            information_array['user_prediction'] = "ham"

        if tweet_score is 1:
            information_array['tweet_prediction'] = "spam"
        else:
            information_array['tweet_prediction'] = "ham"

        if fuzzy_score is 1:
            information_array['fuzzy_score'] = "spam"
        else:
            information_array['fuzzy_score'] = "ham"

        return information_array

    def get_prediction_report(self):
        return self.information_array


if __name__ == '__main__':
    SpamDetector.main()
