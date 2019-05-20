import pickle


class Retrainer:
    data = "data/"
    __check = None
    drifted_tweets_list = []
    tweets_dataset = None

    def __init__(self):
        print("Initialized Retrainer")

    def retrieve_drifted_tweets(self):
        # load the spam tokens
        filename = self.data + "drifted_tweets.p"
        drifted_tweets = pickle.load(open(filename, 'rb'))
        self.drifted_tweets_list = drifted_tweets

    def get_drifted_tweets(self):
        return self.drifted_tweets_list

    # def retrain_tweet_classifier(self):
    #
    # def get_retrain_score(self):
    #
    # def load_tweets_dataset(self):
