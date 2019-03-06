import tweepy

access_token = '250547225-fktvVk30HdHXnLgwhlHM5dBdv63YdrQBIuWjbjtV'
access_token_secret = '7NU8T6AWypW2jT0DyvTvjb9UqFaF1TfIMBeknptRJYHjH'
consumer_key = 'VhzN7FMeTq7lKvwAKV02Ho9Dw'
consumer_secret = 'K8pAy6MTLtMB5R5WSklrR0lqudb75zFkjfHWtscHqj7YYixsRr'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

# def get_user(user_name):

# create class to get twitter user data
user = api.get_user('nottoraretaaka')

print(user)
print(user.followers_count)
# for friend in user.friends():
#    print(friend.screen_name)
