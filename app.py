from flask import Flask, render_template, url_for, request, flash, redirect
from TweetListener import TweetListener
from SpamDetector import SpamDetector
from Retrainer import Retrainer

app = Flask(__name__)

app.config['SECRET_KEY'] = '4551a264s45df5scs541'


@app.route("/")
@app.route("/home")
def home():
    return render_template('home_landing.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
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
                return render_template('full_detection.html', prediction=classification_report)
    else:
        return redirect(url_for('home'))


@app.route("/review", methods=['GET', 'POST'])
def review():
    if request.method == 'POST':
        if 'checked_drift' in request.form:
            drifted_check_tweets = request.form['checked_drift']
            drifted_check_tweets = drifted_check_tweets.split(", ")
            retrain = Retrainer()
            retrain.write_flagged_drifted_tweets(drifted_check_tweets)
            retrain.get_number_flagged_drifted_tweets()
            return render_template('review.html', retrain_information_array='lol')
        else:
            # invalid operation
            flash(f'Operation is invalid', 'danger')
            return redirect(url_for('review'))



    else:
        retrain = Retrainer()
        retrain.retrieve_unflagged_drifted_tweets()
        drifted_tweets = retrain.get_unflagged_drifted_tweets()
        return render_template('review.html', drifted=drifted_tweets)


@app.route("/classify_tweet")
def classify_tweet():
    return render_template('tweet_detection.html')


if __name__ == '__main__':
    app.run(debug=True)
