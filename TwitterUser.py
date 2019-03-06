# Importing custom Twitter API class
import TwitterAuthentication

# create and initialise Twitter API object
TwitterAPI = TwitterAuthentication.TwitterAuthentication()

# authenticate
api = TwitterAPI.authenticate()

user = api.get_user('nottoraretaaka')

print(user)
# print(user.followers_count)
# for friend in user.friends():
#    print(friend.screen_name)
