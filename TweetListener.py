import TwitterAPI


class TweetListener:
    tweet = None

    def __init__(self):
        twitter_api = TwitterAPI.TwitterAPI()
        twitter_api.authenticate()
        print('initialized tweet listener')

    def stream_tweet(self, handle):
        twitter_api = TwitterAPI.TwitterAPI()
        twitter_api.authenticate()
        # fake spam account @HapumalB
        if twitter_api.check_user(handle):
            if handle.lower() == 'hapumalb':
                tweet_object = twitter_api.streamTweetFromUser(handle.lower())
                self.tweet = tweet_object.text
                return tweet_object
            else:
                return "User found"
        else:
            return "User not found"
