# Importing custom Twitter API class
import TwitterAuthentication

# create and initialise Twitter API object
TwitterAPI = TwitterAuthentication.TwitterAuthentication()

# authenticate
api = TwitterAPI.authenticate()

# user = api.get_user('nottoraretaaka')
tweet = api.get_status(id=329857704132751360)
print(tweet.text)
# print(user.followers_count)
# for friend in user.friends():
#    print(friend.screen_name)
