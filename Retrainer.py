import pickle
import os
import csv
from TweetDetectModel import TweetDetectModel


class Retrainer:
    data = "data/"
    __check = None
    drifted_tweets_list = []
    tweets_dataset = None
    get_flagged_drifted_tweets_list = []
    number_flagged_drifted_tweet = None

    def __init__(self):
        print("Initialized Retrainer")

    def retrieve_unflagged_drifted_tweets(self):
        # load the spam tokens
        filename = self.data + "drifted_tweets.p"
        drifted_tweets = pickle.load(open(filename, 'rb'))
        self.drifted_tweets_list = drifted_tweets

    def get_unflagged_drifted_tweets(self):
        return self.drifted_tweets_list

    def write_flagged_drifted_tweets(self, drifted_list):
        exists = os.path.isfile('./data/flagged_drifted_tweets.p')
        if exists:
            # load drifted tweets from file using pickle
            filename = self.data + "flagged_drifted_tweets.p"
            flagged_drifted_tweets = pickle.load(open(filename, 'rb'))
            # add to a list and append the new tweet
            flagged_drifted_tweets.extend(drifted_list)
            # save drifted tweets to file using pickle
            filename = self.data + "flagged_drifted_tweets.p"
            pickle.dump(flagged_drifted_tweets, open(filename, "wb"))
        else:
            # save drifted tweets to file using pickle
            filename = self.data + "flagged_drifted_tweets.p"
            pickle.dump(drifted_list, open(filename, "wb"))

    def get_number_flagged_drifted_tweets(self):
        exists = os.path.isfile('./data/flagged_drifted_tweets.p')
        if exists:
            # load drifted tweets from file using pickle
            filename = self.data + "flagged_drifted_tweets.p"
            flagged_drifted_tweets = pickle.load(open(filename, 'rb'))
            # add to a list and append the new tweet
            length = len(flagged_drifted_tweets)
            self.number_flagged_drifted_tweet = length
            return length
        else:
            self.number_flagged_drifted_tweet = 0
            return 0

    def get_flagged_drifted_tweets(self):
        # load the spam tokens
        flagged_drifted_tweets = []
        exists = os.path.isfile('./data/flagged_drifted_tweets.p')
        if exists:
            filename = self.data + "flagged_drifted_tweets.p"
            flagged_drifted_tweets = pickle.load(open(filename, 'rb'))
            self.drifted_tweets_list = flagged_drifted_tweets
            return flagged_drifted_tweets
        else:
            return flagged_drifted_tweets

    def retrain_information(self):
        retrain_information = dict()

        flagged_drift_tweets = self.get_flagged_drifted_tweets()

        if flagged_drift_tweets is not None and len(flagged_drift_tweets) > 0:
            retrain_information["flagged_drift_tweets"] = flagged_drift_tweets
        else:
            retrain_information["flagged_drift_tweets"] = 'No Flagged Drift Tweets Available'

        number_flagged_tweets = self.get_number_flagged_drifted_tweets()

        if number_flagged_tweets is not None:
            retrain_information["number_drift_tweets"] = number_flagged_tweets

        return retrain_information

    def delete_unflagged_drifted_tweets(self, delete_tweets):

        # load the spam tokens
        filename = self.data + "drifted_tweets.p"
        unflagged_drifted_tweets = pickle.load(open(filename, 'rb'))
        new_unflagged_drifted_tweets = [item for item in unflagged_drifted_tweets if item not in delete_tweets]

        # save drifted tweets to file using pickle
        filename = self.data + "drifted_tweets.p"
        pickle.dump(new_unflagged_drifted_tweets, open(filename, "wb"))

    def delete_flagged_drifted_tweets(self, drifted_check_tweets):

        # load the spam tokens
        filename = self.data + "drifted_tweets.p"
        unflagged_drifted_tweets = pickle.load(open(filename, 'rb'))
        new_unflagged_drifted_tweets = [item for item in unflagged_drifted_tweets if item not in drifted_check_tweets]

        # save drifted tweets to file using pickle
        filename = self.data + "drifted_tweets.p"
        pickle.dump(new_unflagged_drifted_tweets, open(filename, "wb"))
        return new_unflagged_drifted_tweets

    def delete_retrained_flagged_drifted_tweets(self, retrain_tweets):

        # load the spam tokens
        filename = self.data + "flagged_drifted_tweets.p"
        flagged_drifted_tweets = pickle.load(open(filename, 'rb'))
        empty_flagged_drifted_tweets = [item for item in flagged_drifted_tweets if item not in retrain_tweets]

        # save drifted tweets to file using pickle
        filename = self.data + "drifted_tweets.p"
        pickle.dump(empty_flagged_drifted_tweets, open(filename, "wb"))

    def retrain_tweet_classifier(self, retrain_tweets):
        if self.load_add_dataset(retrain_tweets):

            # run the tweet detect model with newly added flagged drifted tweets
            retrain_model = TweetDetectModel()
            if retrain_model.main():
                self.delete_retrained_flagged_drifted_tweets(retrain_tweets)
                return True
            else:
                return False

        else:
            return False

    def load_add_dataset(self, retrain_tweets):
        try:
            # open file containing dataset
            with open('dataset/SpamTweetsFinalDataset.csv', 'a') as fd:
                for tweet in retrain_tweets:
                    tweet = tweet.replace(',', '')
                    tweet_string = tweet + ",spam\n"
                    fd.write(tweet_string)
            # close the file
            fd.close()
            return True

        except Exception as e:
            return False


if __name__ == '__main__':
    retrain = Retrainer()
    lala = retrain.load_add_dataset(
        ["hi how are you and what are you doing these days lalala", "check this out blah blah blah"])
    print(lala)
