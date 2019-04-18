from flask import Flask, render_template, url_for, request
import pickle
import nltk
import TwitterAPI
import re
from nltk.tokenize import word_tokenize
import TweetListener as TweetListener
import SpamDetector as SpamDetector


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        tweet_listen = TweetListener.TweetListener()
        tweetObj = tweet_listen.stream_tweet()



    return render_template('result.html', prediction=prediction_test)


@app.route('/retrieve_classify', methods=['GET', 'POST'])
def retrieve_classify():
    if request.method == 'POST':
        tweet = request.form['tweet']
        prediction_test = "coming soon"
        print("Prediction coming soon")

    return render_template('result.html', prediction=prediction_test)


if __name__ == '__main__':
    app.run(debug=True)
