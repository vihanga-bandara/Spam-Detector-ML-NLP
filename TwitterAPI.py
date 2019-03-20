import tweepy


class TwitterAPI:
    _access_token = '250547225-fktvVk30HdHXnLgwhlHM5dBdv63YdrQBIuWjbjtV'
    _access_token_secret = '7NU8T6AWypW2jT0DyvTvjb9UqFaF1TfIMBeknptRJYHjH'
    _consumer_key = 'VhzN7FMeTq7lKvwAKV02Ho9Dw'
    _consumer_secret = 'K8pAy6MTLtMB5R5WSklrR0lqudb75zFkjfHWtscHqj7YYixsRr'
    _auth = None
    _api = None

    def __init__(self):
        self._auth = tweepy.OAuthHandler(self._consumer_key, self._consumer_secret)
        self._auth.set_access_token(self._access_token, self._access_token_secret)

    def authenticate(self):
        api = tweepy.API(self._auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
        self._api = api
        return api

    def getTweet(self, tweetId):
        # hardcoded tweetId for now
        tweetId = 1107192300692660224
        tweet = self._api.get_status(id=tweetId)
        print(tweet.text)
        return tweet

    def getUser(self, tweetUserName):
        tweetUserName = tweetUserName
        tweetUser = self._api.get_user(tweetUserName)
        print(tweetUser)
        return tweetUser

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
