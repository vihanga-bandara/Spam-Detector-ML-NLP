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
        if len(tweets) == 1:
            return False
        else:
            return True

    def on_error(self, status):
        print(status)

    def disconnect(self):
        self.__stream.disconnect()


class TwitterAPI:
    _access_token = '250547225-fktvVk30HdHXnLgwhlHM5dBdv63YdrQBIuWjbjtV'
    _access_token_secret = '7NU8T6AWypW2jT0DyvTvjb9UqFaF1TfIMBeknptRJYHjH'
    _consumer_key = 'VhzN7FMeTq7lKvwAKV02Ho9Dw'
    _consumer_secret = 'K8pAy6MTLtMB5R5WSklrR0lqudb75zFkjfHWtscHqj7YYixsRr'
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
        # hardcoded tweetId for now
        # tweetId = 1107192300692660224
        tweet = self._api.get_status(id=tweetId)
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
            tweetUser = self._api.get_user(handle)
        except tweepy.TweepError as e:
            if e.api_code is 50:
                user_exist = False

        return user_exist

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

        twitterStream = Stream(self._auth, Listener())
        print("Listening to incoming tweets")
        twitterStream.filter(follow=[username])
        print(tweets[0].text)
        print("Listener has disconnected")
        tweet = tweets[0]
        return tweet

    def rate_limit(self):
        self._rate_limit = self._api.rate_limit_status()
        return self._rate_limit


if __name__ == '__main__':
    twitter_api = TwitterAPI()
    twitter_api.authenticate()
    twitter_api.check_user("hapumalbbb")

    exit(0)
