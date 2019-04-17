import TwitterAPI

twitterAPI = TwitterAPI.TwitterAPI()
twitterAPI.authenticate()
twitterAPI.streamUser('1105567862419324936')
