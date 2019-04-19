from flask import Flask, render_template, url_for, request
import TweetListener as TweetListener
from SpamDetector import SpamDetector
from TweetListener import TweetListener

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        spam_detector = SpamDetector()
        spam_detector.main(None, tweet, None)
        classification_report = spam_detector.get_prediction_report()
        return render_template('result_tweet_only.html', prediction=classification_report)


# @app.route('/retrieve_classify', methods=['GET', 'POST'])
# def retrieve_classify():
#     if request.method == 'POST':
#         tweet_listen = TweetListener()
#         tweet_obj = tweet_listen.stream_tweet()
#         spam_detector = SpamDetector()
#         spam_detector.main("checking for spam drift", None, None)
#         classification_report = spam_detector.get_prediction_report()
#
#     return render_template('result.html', prediction=classification_report)


if __name__ == '__main__':
    app.run(debug=True)
