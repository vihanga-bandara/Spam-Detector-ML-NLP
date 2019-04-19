#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:52:53 2019

@author: vihanga123
"""
import numpy as np
import pandas as pd
from Preprocessor import Preprocessor
from SpamDictionary import SpamDictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from pyjarowinkler import distance
from datamuse import datamuse
import pickle
import textdistance


class DriftDetector:
    pickle = "pickle/"
    dataset = "dataset/"
    preprocessor = Preprocessor()

    def __init__(self):
        print('Initializing Drift Detector')

    # function to get score of each feature
    def display_scores(self, vectorizer, tfidf_result):
        scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tfidf_result.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        spam_tokens = []
        for item in sorted_scores:
            print("{0:50} Score: {1}".format(item[0], item[1]))
            if type(item[0]) == str and item[1] > 1.8:
                spam_tokens.append(item[0])
                print('adding spam token - {0}'.format(item[0]))

        ham_words = ['3', 'part', 'know', 'nikkis', 'beacon', 'advisory', 'banks', 'direct', 'guide', 'way', 'maverick',
                     'feeling', 'pressure', 'probably', 'valued', 'seven', 'could', 'relief', 'website', 'effective',
                     'taught', 'pack', 'oz', 'beholding', 'one', 'anyone', 'read', 'sassy', 'tea', 'seriously', 'list']
        for word in list(spam_tokens):
            if word in ham_words:
                spam_tokens.remove(word)
        # save spam tokens using pickle
        filename = "dataset/spam_tokens.p"
        pickle.dump(spam_tokens, open(filename, "wb"))
        return spam_tokens

    def calculate_score(self, sc, num_words):
        score = (sc / num_words) * 100
        return score

    def tweet_token_analogy_alg(self, tweet_tokens, spam_tokens):
        """ This function will retrieve all similar and related words from twinword and datamuse API
            for each tweet token and then will compare it with each spam token using jaro-winkler algorithm.
            If it has a matching counterpart that particular tweet token will be given a score. After the whole process is
            completed, average weighted score will be calculated and then returned"""

        tweet_score = 0

        # initialize words list
        words = []
        for token in tweet_tokens:
            # using twinword api get associative words
            # response = requests.get("https://twinword-word-associations-v1.p.rapidapi.com/associations/?entry=token",
            #                 headers={
            #                         "X-RapidAPI-Key": "5557d82ac7msheb9fc00a6b39b02p1f5141jsn4a6fd56d88ce"
            #                     }
            #                     )
            # print(response)

            # with open('response.json', 'w') as outfile:
            #     json.dump(response.text, outfile)
            # filter it according to score

            # # using datamuse api get related words
            api = datamuse.Datamuse()
            datamuse_response_similar = api.words(ml=token, max=5)
            print('Getting related analogies for tweet token...')
            # no need to check for score since we take only max 5
            print(datamuse_response_similar)
            # contains related word from datamuse API
            datamuse_list = []
            for response in datamuse_response_similar:
                datamuse_list.append(response['word'])
                words.append(response['word'])

            # using datamuse api get related terms
            response2 = api.words(rel_syn=token, max=3)
            print('Getting similar words for spam token...')
            # no need to check for score since we take only max 5
            print(response2)
            for response in response2:
                words.append(response['word'])

            """if word does not have a synonym check the json response for a result code of 462, 
            if so can try to correct spellings and get using datamuse api--TODO"""

            # with open('response.json') as json_file:
            #     data = json.load(json_file)

            # compare both the lists to identify duplicates and then remove them and get only the top 5 or 10 words

            # get associative words for each token
            # similiar_tokens_json = json.loads(data)
            # print(similiar_tokens_json)
            # print('Getting associated analogies for tweet token...')
            # twinword_dict = similiar_tokens_json['associations_scored']

            # contains associative words from twinword API
            # twinword_list = []
            # # get tokens which are above 80 score from
            # for token_name, value in twinword_dict.items():
            #     if value > 80:
            #         twinword_list.append(token_name)
            #         words.append(token_name)

            found_similiar_bool = False
            print('Searching for tokens...')
            for simtoken in words:
                for sptoken in spam_tokens:
                    print('Searching for similar word - {0} and spam token - {1}...'.format(simtoken, sptoken))
                    word1, word2 = simtoken, sptoken
                    similarity1 = distance.get_jaro_distance(word1, word2, winkler=True, scaling=0.1)
                    similarity2 = textdistance.jaro_winkler(word1, word2)
                    similarity = (similarity1 + similarity2) / 2
                    print(similarity)
                    if similarity > 0.94:
                        tweet_score += 1
                        found_similiar_bool = True
                        print('Found identical or matching token for  {0}'.format(token))
                        break
                    else:
                        print('similar word - {0} and spam token - {1}...No Match'.format(simtoken, sptoken))
                if found_similiar_bool:
                    break
                else:
                    print('Searching for spam tokens in similar word - {0}...Not Found'.format(simtoken))

            if found_similiar_bool is False:
                print('No similar tokens that maps to spam token were found for tweet token - {0}'.format(token))
                continue

        # calculate average weighted score
        score = self.calculate_score(tweet_score, len(tweet_tokens))
        return score

    def spam_token_analogy_alg(self, tweet_tokens, spam_tokens):
        """ This function will retrieve all similar and related words from datamuse API
            for each spam token and then will compare it with each tweet token using jaro-winkler algorithm.
            If it has a matching counterpart that particular tweet token will be given a score. After the whole process is
            completed, average weighted score will be calculated and then returned"""

        # initialize score
        tweet_score = 0

        # invoke local spam dictionary class
        spam_dict = SpamDictionary()

        # get datamuse words for each spam_token
        for tweet_token in tweet_tokens:
            # boolean to check if identical token is found
            found_similiar_bool = False
            for spam_token in spam_tokens:
                # use spam dictionary to get terms
                words = spam_dict.get_words_per_spam_token(spam_token)
                print('Searching for tokens...')
                for similar_spam_token in words:
                    # print('Searching for similar spam token - {0} and tweet token - {1}...'.format(similar_spam_token,
                    #                                                                                tweet_token))
                    word1, word2 = similar_spam_token, tweet_token
                    similarity1 = distance.get_jaro_distance(word1, word2, winkler=True, scaling=0.1)
                    similarity2 = textdistance.jaro_winkler(word1, word2)
                    similarity = (similarity1 + similarity2) / 2
                    if similarity > 0.9:
                        tweet_score += 1
                        found_similiar_bool = True
                        print(
                            'Found identical or matching similar spam token for  tweet token - {0}'.format(tweet_token))
                        break
                if found_similiar_bool:
                    break

            if found_similiar_bool:
                continue
            else:
                print('No spam tokens found for tweet token {0}'.format(tweet_token))

        # calculate score
        score = self.calculate_score(tweet_score, len(tweet_tokens))
        return score

    def kmeans_unsupervised_predict(self, tweet):
        """ Final check to see if the tweet has spam intent using unsupervised model
            to check if it is in the correct cluster, will return 0 if not spam and will
            return 1 if it is spam"""

        import pickle

        # load tfidf vectorizer from directory
        filename = self.pickle + "Unsupervised_Vectorizer_TFIDF.p"
        vectorizer = pickle.load(open(filename, 'rb'))

        # load the Unsupervied KMeans Model from directory
        filename = self.pickle + 'Unsupervised_KMeans_Model.sav'
        model = pickle.load(open(filename, 'rb'))

        X = vectorizer.transform(['get free twitter followers'])
        predicted = model.predict(X)
        print(predicted)

        if predicted == 0:
            return 1
        else:
            return 0

    def predict(self, tweet_obj, check):

        if check is 0 or check is 2:
            # preprocess tweet
            processed_tweet = self.preprocessor.preprocess_tweet(tweet_obj.text)
        else:
            processed_tweet = self.preprocessor.preprocess_tweet(tweet_obj)

        tweet_tokens = nltk.word_tokenize(processed_tweet)

        # tokenize those words and add it to a list
        # load the dataset
        filename = self.dataset + 'SpamTweetsFinalDataset.csv'
        df = pd.read_csv(filename, header=None)

        # print general information about the dataset that is loaded
        print(df.info())
        print(df[1])

        # using only the spam labelled data
        spam_tweets = df.loc[df[1] == 'spam']
        spam_tweets = self.preprocessor.preprocess_spam_tweets(spam_tweets[0])

        # dummy function to avoid preprocessing and tokenization in tfid
        def dummy_fun(doc):
            return doc

        # initialize TF-ID vectorizer
        tfidf = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            token_pattern=None)

        tfidf_result = tfidf.fit_transform(spam_tweets)
        # print features using the document matrix
        print(tfidf.get_feature_names())

        """Get only the most important words top 50 scores maybe"""

        spam_tokens = self.display_scores(tfidf, tfidf_result)
        print(spam_tokens)
        print(tweet_tokens)

        # run first drift check
        first_score, second_score, unsupervised_score = 0, 0, 0
        first_score = self.tweet_token_analogy_alg(tweet_tokens, spam_tokens)
        if first_score >= 40:
            print('This tweet might be spam therefore it will be sent for reporting. Percentage - {0}%'.format(
                first_score))
            # classify as maybe spam and send it to admin panel
            # add tweet to new drifted tweets file

        else:
            # run second drift check
            second_score = self.spam_token_analogy_alg(tweet_tokens, spam_tokens)

        if second_score >= 40:
            print('This tweet might be spam therefore it will be sent for reporting. Percentage - {0}%'.format(
                second_score))
        else:
            unsupervised_score = self.kmeans_unsupervised_predict(tweet_tokens)

        if unsupervised_score == 1:
            print('This tweet might be spam therefore it will be sent for reporting.')
        elif unsupervised_score == 0 and first_score < 30 and second_score < 30:
            print('This tweet is not spam but user can manually report it for spam')

        # else the tweet will be shown and it will have the ability to be reported manually
        # and then labelled accordingly by the admin

        # if it does work out and if the tweet is recognised as spam using either
        # function 1 or function 2 or unsupervised model
        # we wont retrain it until the admin see it and manually verifies that it is
        # spam or not. This is done to ensure accuracy in the model
        # and to reduce false positives

        # manually reported tweets will have the ability to be checked by
        # user and then manually labelled. When the detector accuracy reduces by 80%
        # it will take all the reported tweets that are labelled spam and then add it to dataset and retrained

        # have an option where you could give tweets randomly and allow users
        # label them. If it contains tweets which arent really spam now.
        # if many people report it as not spam. Those data will be removed from the
        # dataset to make the retraining faster and efficient

        # and make the model accuracy higher and precise
