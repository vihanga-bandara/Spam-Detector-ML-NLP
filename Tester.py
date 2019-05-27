from TweetDetectModel import TweetDetectModel
from Classifiers import UserClassifier
from SpamFuzzyController import SpamFuzzyController
from DriftDetector import DriftDetector
from SpamDetector import SpamDetector


class Tester:

    def tweet_detector_tests(self):
        # tests run on the tweet detector
        detector = TweetDetectModel()
        test_array_thesis_data = ['HOW DO YOU GET UNLIMITED FREE TWITTER FOLLOWERS? http://tinyurl.com/3xkr5hc',
                                  'free twitter followers is the choice that i have and i know it',
                                  'Want thousands of people to follow you for free?',
                                  'Special free offers EXTRA 6% Off on Gold and Silver coins',
                                  'Im going back to your Genius Bar to complain. #annoyed',
                                  'I am suing my insurance company and you just managed to make it to the top of my shit list. #Ineedthosepictures',
                                  'Click to check your daily and become rich',
                                  'Here is a small gift for you #gifts',
                                  'Best investment from us, retweet to win',
                                  'Obtain complimentary coin, check it out now']

        test_array_real_data = [
            'GET MORE FOLLOWERS FREE HERE. GET UP TO 70 FOLLOWERS -> http://poin.pixub.com <- #OPENFOLLOW #JFB',
            'Auto followers no spam! http://poin.pixub.com (ada limit untuk menghindari suspended) #OPENFOLLOW #JFB',
            'GET 70 FOLLOWERS http://poin.pixub.com #JFB',
            'GRATIS! Auto followers no spam! http://poin.pixub.com (ada limit untuk menghindari suspended) #OPENFOLLOW #JFB',
            'FREE Instagram followers ♛ Click ☟ http://followerzforfree.website/',
            'Save to 30% Off Your Order, $1.99 Hosting, $2.95 .Coms - http://promocodeus.net/stores/godaddy-promo-codes/ … #godaddy @godaddycouponus',
            'Super hyped about Bluestone Capital providing the support and backing to drive Esports in Sri Lanka and beyond. With the shared passion and commitment of #TeamGLK, we are excited for what lies in the future for Esports! #lka #Esports #gamerlk',
            'Police have arrested a 32 year old suspect (H Pawithra Madushanka) over the 13 hand grenades found in a school in Baduraliya. #SriLanka #lka',
            'Some trying to spread fake news UN deployed in #SriLanka. Total and absolute lies. I checked with CDS who told me these are dated photos of armored personal carriers prepared to be sent on UN peace keeping missions in Mali; perhaps on exercise w our troops.']

        print('\n*** Test data ***\n')

        for test in test_array_thesis_data:
            # run each test and classify tweets
            res = detector.classify(test)
            print(res)

        print('\n*** Real data ***\n')

        for test in test_array_real_data:
            # run each test and classify tweets
            res = detector.classify(test)
            print(res)

    def user_detector_tests(self):
        detector = UserClassifier()
        detector.classify_user_name("rameshliyanage")
        getprobval = detector.get_proba_value()
        getprobscore = detector.get_prediction_score()
        getpredtype = detector.get_prediction_type()

        print("{} - Probability Value | {} - Probability Score | {} - Prediction Type".format(getprobval, getprobscore,
                                                                                              getpredtype))

    def spam_fuzzy_controller_tests(self):
        spamfuz = SpamFuzzyController()
        spamfuz.fuzzy_initialize()
        spamfuz.fuzzy_predict(40, 40)
        spamfuz.fuzzy_predict(50, 50)
        spamfuz.fuzzy_predict(10, 10)
        spamfuz.fuzzy_predict(40, 60)
        spamfuz.fuzzy_predict(60, 60)
        spamfuz.fuzzy_predict(80, 80)
        spamfuz.fuzzy_predict(90, 90)
        spamfuz.fuzzy_predict(20, 90)
        spamfuz.fuzzy_predict(40, 90)
        spamfuz.fuzzy_predict(90, 20)
        spamfuz.fuzzy_predict(90, 40)
        spamfuz.fuzzy_predict(45, 70)
        spamfuz.fuzzy_predict(70, 60)

    def drift_detector_tests(self):
        drift_detector = DriftDetector()

        tweets = ['Click to check your daily and become rich',
                  'Here is a small gift for you #gifts',
                  'Best investment from us, retweet to win',
                  'Obtain complimentary coin, check it out now']

        for tweet in tweets:
            # run each test and classify tweets
            report = drift_detector.predict(tweet, 1)
            print(" Spam Status - {0} | Spam Score - {1} | Tweet - {2} ".format(report["spam_status"],
                                                                                report["spam_score"], report["tweet"]))

    def system_tests(self):
        # full flow tests
        # basic tweet entered identified as spam with user score, tweet score, fuzzy score
        spam_detector = SpamDetector()


if __name__ == '__main__':
    run_tests = Tester()
    run_tests.tweet_detector_tests()
    # run_tests.user_detector_tests()
    # run_tests.spam_fuzzy_controller_tests()
    # run_tests.spam_fuzzy_controller_tests()
