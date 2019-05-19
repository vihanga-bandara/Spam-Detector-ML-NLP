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
        check_user, tweet_user_object = twitter_api.check_user(handle)
        if check_user:
            if handle.lower() == 'hapumalb':
                tweet_object = twitter_api.streamTweetFromUser(handle.lower())
                self.tweet = tweet_object.text
                return tweet_object
            else:
                tweet_object = twitter_api.get_latest_tweet(tweet_user_object.screen_name)
                return tweet_object
        else:
            return "User not found"
