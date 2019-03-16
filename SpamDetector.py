import pickle

# load the SpamTweetDetectModel word_features from directory
filename = "wordfeatures.p"
word_features = pickle.load(open(filename, 'rb'))

# load the SpamTweetDetectModel from directory
filename = "SpamTweetDetectModel.sav"
nltk_ensemble = pickle.load(open(filename, 'rb'))
