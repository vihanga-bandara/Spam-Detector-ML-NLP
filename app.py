from flask import Flask, render_template, url_for, request, flash, redirect
from TweetListener import TweetListener
from SpamDetector import SpamDetector
from Retrainer import Retrainer

app = Flask(__name__)

# secret key needed to initiate flash messages
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
                # call spam detector instance
                spam_detector = SpamDetector()
                # send tweet object to spam detector instance
                spam_detector.main(tweet_obj, 1)
                # generate classification report
                classification_report = spam_detector.get_prediction_report()
                return render_template('full_detection.html', prediction=classification_report)
    else:
        return redirect(url_for('home'))


@app.route("/review", methods=['GET', 'POST'])
def review():
    if request.method == 'POST':
        # check whether checkbox has been checked and data exists
        if 'checked_drift' in request.form or 'retrain' in request.form:
            # instantiate retrainer object
            retrain = Retrainer()
            # check the type of submit button that was pressed (confirm spam, delete useless spam, retrain)
            if 'confirm_spam' in request.form:
                # get post form data
                drifted_check_tweets = request.form.getlist('checked_drift')
                # write the flagged tweets by the sysadmin to pickle
                retrain.write_flagged_drifted_tweets(drifted_check_tweets)
                # generate retrain information details
                retrain_information_array = retrain.retrain_information()
                # delete the tweets that have been flagged already by the admin
                drifted_tweets = retrain.delete_flagged_drifted_tweets(drifted_check_tweets)
                # successfully written drifted tweets
                flash(f'Flagged Tweets Successfully', 'success')
                return render_template('review.html', retrain_information_array=retrain_information_array,
                                       drifted=drifted_tweets)
            elif 'remove' in request.form:
                # get post form data
                delete_tweets = request.form.getlist('checked_drift')
                # delete tweets flagged as not spam from pickle
                retrain.delete_unflagged_drifted_tweets(delete_tweets)
                flash(f'Deleted Tweets Successfully', 'success')
                return redirect(url_for('review'))

            elif 'retrain' in request.form:
                # retrieve all tweets flagged by user for retraining
                retrain_tweets = retrain.get_flagged_drifted_tweets()
                # add the newly identified spammy tweets and retrain the tweet classifier
                if retrain.retrain_tweet_classifier(retrain_tweets):
                    # successfully retrained path
                    flash(f'Successfully Retrained Classifier', 'success')
                    # generate retrain information details
                    retrain_information_array = retrain.retrain_information()
                    # generate un flagged tweets
                    retrain.retrieve_unflagged_drifted_tweets()
                    # retrieve drifted tweets
                    drifted_tweets = retrain.get_unflagged_drifted_tweets()
                    return render_template('review.html', retrain_information_array=retrain_information_array,
                                           drifted=drifted_tweets)
                else:
                    # unsuccessful path
                    flash(f'Retraining not successfully, Please Try again', 'danger')
                    return redirect(url_for('review'))


        else:
            # invalid operation
            flash(f'Operation is invalid', 'danger')
            return redirect(url_for('review'))

    else:
        # normal path without any post or get method involved
        # instantiate retrainer object
        retrain = Retrainer()
        # generate tweets that have been identified as drifted
        retrain.retrieve_unflagged_drifted_tweets()
        # retrieve drifted tweets
        drifted_tweets = retrain.get_unflagged_drifted_tweets()
        # generate retrain information details
        retrain_information_array = retrain.retrain_information()
        return render_template('review.html', retrain_information_array=retrain_information_array,
                               drifted=drifted_tweets)


@app.route("/classify_tweet")
def classify_tweet():
    return render_template('tweet_detection.html')


@app.errorhandler(Exception)
def handle_error(e):
    return redirect(url_for('error'))


@app.route("/error")
def error():
    return render_template('error.html')


@app.route("/hello")
def hello():
    return render_template('extra.html')


if __name__ == '__main__':
    app.run(debug=True)
