import TwitterAPI


class TweetListener():
    tweet = None

    def __init__(self):
        twitter_api = TwitterAPI.TwitterAPI()
        twitter_api.authenticate()
        print('initialized tweet listener')

    def stream_tweet(self):
        twitter_api = TwitterAPI.TwitterAPI()
        twitter_api.authenticate()
        # fake spam account @HapumalB
        tweet_object = twitter_api.streamTweetFromUser('1105567862419324936')
        self.tweet = tweet_object.text
        return tweet_object
