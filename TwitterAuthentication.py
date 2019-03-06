import tweepy


class TwitterAuthentication:
    _access_token = '250547225-fktvVk30HdHXnLgwhlHM5dBdv63YdrQBIuWjbjtV'
    _access_token_secret = '7NU8T6AWypW2jT0DyvTvjb9UqFaF1TfIMBeknptRJYHjH'
    _consumer_key = 'VhzN7FMeTq7lKvwAKV02Ho9Dw'
    _consumer_secret = 'K8pAy6MTLtMB5R5WSklrR0lqudb75zFkjfHWtscHqj7YYixsRr'
    _auth = None

    def __init__(self):
        self._auth = tweepy.OAuthHandler(self._consumer_key, self._consumer_secret)
        self._auth.set_access_token(self._access_token, self._access_token_secret)

    def authenticate(self):
        api = tweepy.API(self._auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
        return api
