from flask import Flask, render_template, url_for, request, flash, redirect
from TweetListener import TweetListener
from SpamDetector import SpamDetector
from TwitterAPI import TwitterAPI

app = Flask(__name__)

app.config['SECRET_KEY'] = '4551a264s45df5scs541'

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 19, 2018'
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home_landing.html', posts=posts)


@app.route('/review', methods=['GET', 'POST'])
def review():
    if request.method == 'POST':
        if request.method == 'POST':
            # initialize Tweet Listener
            tweet_listen = TweetListener()
            # check twitter handle and listen
            user_handle = request.form['twitter_handle']
            tweet_obj = tweet_listen.stream_tweet(user_handle)
            # check if custom user or normal user or invalid user
            if isinstance(tweet_obj, str):
                # invalid user
                flash(f'Handle entered is not a valid user', 'danger')
                return redirect(url_for('home'))

            elif isinstance(tweet_obj, object):
                # custom user or normal user
                spam_detector = SpamDetector()
                spam_detector.main(tweet_obj, 1)
                classification_report = spam_detector.get_prediction_report()
                return render_template('review.html', prediction=classification_report)


@app.route("/classify")
def classify():
    return render_template('full_detection.html')


@app.route("/classify_tweet")
def classify_tweet():
    return render_template('tweet_detection.html')


if __name__ == '__main__':
    app.run(debug=True)
