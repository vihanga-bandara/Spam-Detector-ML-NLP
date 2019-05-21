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
        list = request.form.to_dict(flat=False)
        if 'checked_drift' in request.form:
            retrain = Retrainer()
            if 'confirm_spam' in request.form:
                drifted_check_tweets = request.form.getlist('checked_drift')
                retrain.write_flagged_drifted_tweets(drifted_check_tweets)
                retrain_information_array = retrain.retrain_information()
                drifted_tweets = retrain.delete_flagged_drifted_tweets(drifted_check_tweets)
                # successfully written drifted tweets
                flash(f'Flagged Tweets Successfully', 'success')
                return render_template('review.html', retrain_information_array=retrain_information_array,
                                       drifted=drifted_tweets)
            elif 'remove' in request.form:
                delete_tweets = request.form.getlist('checked_drift')
                retrain.delete_unflagged_drifted_tweets(delete_tweets)
                # successfully writted drifted tweets
                flash(f'Deleted Tweets Successfully', 'success')
                return redirect(url_for('review'))

            elif 'retrain' in request.form:
                retrain_tweets = retrain.get_flagged_drifted_tweets()
                retrain.retrain_tweet_classifier(retrain_tweets)


        else:
            # invalid operation
            flash(f'Operation is invalid', 'danger')
            return redirect(url_for('review'))

    else:
        retrain = Retrainer()
        retrain.retrieve_unflagged_drifted_tweets()
        drifted_tweets = retrain.get_unflagged_drifted_tweets()
        retrain_information_array = retrain.retrain_information()
        return render_template('review.html', retrain_information_array=retrain_information_array,
                               drifted=drifted_tweets)


@app.route("/classify_tweet")
def classify_tweet():
    return render_template('tweet_detection.html')


if __name__ == '__main__':
    app.run(debug=True)
