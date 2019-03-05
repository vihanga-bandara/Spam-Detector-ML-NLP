# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the tweepy library
import tweepy

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = 'YOUR ACCESS TOKEN"'
ACCESS_SECRET = 'YOUR ACCESS TOKEN SECRET'
CONSUMER_KEY = 'YOUR API KEY'
CONSUMER_SECRET = 'ENTER YOUR API SECRET'

# Setup tweepy to authenticate with Twitter credentials:

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

# Create the api to connect to twitter with your creadentials
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
# ---------------------------------------------------------------------------------------------------------------------
# wait_on_rate_limit= True;  will make the api to automatically wait for rate limits to replenish
# wait_on_rate_limit_notify= Ture;  will make the api  to print a notification when Tweepyis waiting for rate limits to replenish
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# The following loop will print most recent statuses, including retweets, posted by the authenticating user and that user’s friends.
# This is the equivalent of /timeline/home on the Web.
# ---------------------------------------------------------------------------------------------------------------------

for status in tweepy.Cursor(api.home_timeline).items(200):
    print(status._json)

# ---------------------------------------------------------------------------------------------------------------------
# Twitter API development use pagination for Iterating through timelines, user lists, direct messages, etc.
# To help make pagination easier and Tweepy has the Cursor object.
# ---------------------------------------------------------------------------------------------------------------------
