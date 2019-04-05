import re
import nltk
from nltk.tokenize import word_tokenize


class Preprocessing:

    def __init__(self):
        print('initialized preprocessor')

    def preprocess_tweet(self, tweet):

        # using regex to identify different combinations in the tweet

        # replacing email addresses with 'emailaddr'
        processed = tweet.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

        # replacing money symbol with 'moneysymb'
        processed = processed.replace(r'$', 'moneysymbol')

        # replacing normal numbers with numbers
        # some not working might need to manually remove
        processed = processed.replace(r'^(\+|-)?\d+$', 'numbr')

        # replaces whitespaces between terms with single space
        processed = processed.replace(r'\s+', ' ')

        # removing leading and trailing whitespaces
        processed = processed.replace(r'^s+|\s+?$', '')

        # try hashtagsss as well which can be a new feature
        # remove hashtags for now
        processed = processed.replace('#', '')

        # remove @handle mentioning names in the tweet
        processed = re.sub(r'@[A-Za-z0-9]+', '', processed, flags=re.MULTILINE)

        # change all letters to lowercase
        processed = processed.lower()

        # remove ASCII character
        #     initial_str = 'Some text ðŸ‘‰ðŸ‘ŒðŸ’¦âœ¨ and some more text'
        #     clean_str = ''.join([c for c in initial_str if ord(c) < 128])
        #     print(clean_str)  # Some text  and some more text

        # remove specific character from the tweet
        cleansed = False
        while not cleansed:
            if '\n' in processed:
                print("Removing next line from words")
                processed = processed.replace("\n", " ")

            elif "-&gt;" in processed:
                print("Removing special characters from words")
                processed = processed.replace("-&gt;", " ")

            elif "&lt;-" in processed:
                print("Removing special characters form words")
                processed = processed.replace("&lt;-", " ")

            else:
                print("Formatting done")
                cleansed = True
        processed = processed

        # replacing links / web addresses with 'webaddr'
        # processed = processed.replace(
        #     r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'webaddr')

        processed = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'webaddr',
                           processed, flags=re.MULTILINE)

        # remove punctuation
        # processed = processed.replace(r'[^\w\d\s]', ' ')
        processed = re.sub(r'[^\w\d\s]', '', processed, flags=re.MULTILINE)

        # remove stop words or useless meaningless words from the tweets
        nltk.download('stopwords')
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))

        tokens = word_tokenize(processed)
        processed = [word for word in tokens if word not in stop_words]
        tweet = " ".join(processed)
        #
        # # using a Porter stemmer to remove word stems
        # ps = nltk.PorterStemmer()
        # processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
        return tweet

    def preprocess_spam_tweets(self, spam_tweets):

        # store the twitter data
        tweets = spam_tweets.str.strip()

        # using regex to identify different combinations in the tweet

        # replacing email addresses with 'emailaddr'
        processed = tweets.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

        # replacing links / web addresses with 'webaddr'
        processed = processed.str.replace(
            r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', 'webaddr')

        # replacing money symbol with 'moneysymb'
        processed = processed.str.replace(r'$', 'moneysymbol')

        # replacing normal numbers with numbers
        # some not working might need to manually remove
        processed = processed.str.replace(r'^(\+|-)?\d+$', 'numbr')

        # remove punctuation
        processed = processed.str.replace(r'[^\w\d\s]', ' ')

        # replaces whitespaces between terms with single space
        processed = processed.str.replace(r'\s+', ' ')

        # removing leading and trailing whitespaces
        processed = processed.str.replace(r'^s+|\s+?$', '')

        # try hashtagsss as well which can be a new feature
        # remove hashtags for now
        processed = processed.replace('#', '')

        # removing @handle from tweets
        # remove @handle mentioning names in the tweet
        # processed = re.sub(r'@[A-Za-z0-9]+', '', processed, flags=re.MULTILINE)

        # change all letters to lowercase
        processed = processed.str.lower()

        # # remove specific character from the tweet - Convert it to Array use first
        # cleansed = False
        # while not cleansed:
        #     if '\n' in processed:
        #         print("Removing next line from words")
        #         processed = processed.replace("\n", " ")
        #
        #     elif "-&gt;" in processed:
        #         print("Removing special characters from words")
        #         processed = processed.replace("-&gt;", " ")
        #
        #     elif "&lt;-" in processed:
        #         print("Removing special characters form words")
        #         processed = processed.replace("&lt;-", " ")
        #
        #     else:
        #         print("Formatting done")
        #         cleansed = True
        # processed = processed

        # remove stop words or useless meaningless words from the tweets

        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))

        processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

        # using a Porter stemmer to remove word stems
        # ps = nltk.PorterStemmer()
        # processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

        # store as list of lists of words
        sentences_ted = []
        for sent_str in processed:
            tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
            sentences_ted.append(tokens)

        return sentences_ted
