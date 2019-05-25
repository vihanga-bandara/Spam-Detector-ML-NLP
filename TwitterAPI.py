import tweepy
from tweepy.streaming import StreamListener
from tweepy import Stream
import time
import requests
import json

tweets = []


class Listener(StreamListener):
    _status = None

    def on_status(self, status):
        tweets.append(status)
        if len(tweets) > 0:
            return False

    def on_error(self, status):
        print(status)

    def on_limit(self, status):
        print("Rate Limit Exceeded, Sleep for 15 Mins")
        time.sleep(15 * 60)
        return True


class TwitterAPI:
    # """ Main account token keys """
    # _access_token = '250547225-fktvVk30HdHXnLgwhlHM5dBdv63YdrQBIuWjbjtV'
    # _access_token_secret = '7NU8T6AWypW2jT0DyvTvjb9UqFaF1TfIMBeknptRJYHjH'
    # _consumer_key = 'VhzN7FMeTq7lKvwAKV02Ho9Dw'
    # _consumer_secret = 'K8pAy6MTLtMB5R5WSklrR0lqudb75zFkjfHWtscHqj7YYixsRr'

    """ Fake account token keys """

    _access_token = '1105567862419324936-tExRyZjEBSisoB3VnRsc1qkM912hbO'
    _access_token_secret = 'lNnifwjeE56G2310gLLFj3CTZznsfTA69zX7LIoHR9XNk'
    _consumer_key = 'aU0zQ6PqESrzxmq7rbvzbTD3M'
    _consumer_secret = 'WYRhPGSjD5btgzvS1Ro2CZhSld5p7wWAiV1dmGZ7mawaFFhVC9'

    _auth = None
    _api = None
    _rate_limit = None

    def __init__(self):
        self._auth = tweepy.OAuthHandler(self._consumer_key, self._consumer_secret)
        self._auth.set_access_token(self._access_token, self._access_token_secret)
        print('initialized twitter api connection')

    def authenticate(self):
        api = tweepy.API(self._auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
        self._api = api
        return api

    def getTweet(self, tweetId):
        if isinstance(tweetId, int):
            tweet = self._api.get_status(id=tweetId)
        else:
            tweet = self._api.get_status(screen_name=tweetId)
        print(tweet.text)
        return tweet

    def getUser(self, tweetUserName):
        tweetUserName = tweetUserName
        tweetUser = self._api.get_user(tweetUserName)
        print(tweetUser)
        return tweetUser

    def check_user(self, handle):
        # url = "https://twitter.com/users/username_available?username={0}".format(handle)
        # response = requests.post(url)
        user_exist = True
        try:
            tweet_user_object = self._api.get_user(handle)
        except tweepy.TweepError as e:
            if e.api_code is 50:
                user_exist = False
                tweet_user_object = None

        return user_exist, tweet_user_object

    def getTweetList(self, tweetLevel):
        if tweetLevel == 0:
            # load file that contains ids of knows spam tweets
            print("return list of all known spam tweetIds")

        elif tweetLevel == 1:
            # load file that contains ids of knows normal tweets
            print("return list of all normal tweetIds")

        else:
            # load file that contains ids of knows normal tweets
            print("return list of all tweetIds")

    def findTweetUser(self, tweetObj):
        # load file that contains ids of knows normal tweets
        tweetUser = tweetObj.user.screen_name
        return tweetUser

    def streamTweetFromUser(self, username):

        # phrases = [“python”, “excel”, “pyxll”]
        # listener = TwitterListener(phrases)

        twitter_stream = Stream(self._auth, Listener())
        print("Listening to incoming tweets")
        user_object = self.getUser(username)
        twitter_stream.filter(follow=[str(user_object.id)])
        latest_index = len(tweets) - 1
        print(tweets[latest_index].text)
        print("Listener has disconnected")
        tweet = tweets[latest_index]
        twitter_stream.disconnect()
        return tweet

    def rate_limit(self):
        self._rate_limit = self._api.rate_limit_status()
        return self._rate_limit

    def get_latest_tweet(self, handle):
        tweet_object = self._api.user_timeline(screen_name=handle.lower(), count=1)[0]
        return tweet_object


if __name__ == '__main__':
    # twitter_api = TwitterAPI()
    # twitter_api.authenticate()
    # user = twitter_api.getUser("hapumalb")
    exit(0)
