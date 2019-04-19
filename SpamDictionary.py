import pickle
from datamuse import datamuse


class SpamDictionary:
    pickle = "pickle/"
    dataset = "dataset/"

    def __init__(self):
        print("Initialized Dictionary")

    # def update_dictionary(self, tokens):

    def retrieve_data_muse(self):
        # load the spam tokens
        filename = self.dataset + "spam_tokens.p"
        spam_tokens = pickle.load(open(filename, 'rb'))

        # load the api
        api = datamuse.Datamuse()
        spam_dictionary = dict()
        for spam_token in spam_tokens:
            words = []
            # contains terms with meaning similar to token from data muse API
            print('Getting related analogies for tweet token...')
            datamuse_response_similar = api.words(ml=spam_token, max=3)
            # no need to check for score since we take only max 5
            print(datamuse_response_similar)
            for response in datamuse_response_similar:
                words.append(response['word'])

            # contains terms with same wordNet synset from data muse API
            print('Getting similar words for spam token...')
            datamuse_response_rel_syn = api.words(rel_syn=spam_token, max=2)
            # no need to check for score since we take only max 5
            print(datamuse_response_rel_syn)
            for response in datamuse_response_rel_syn:
                words.append(response['word'])

            # add the token related terms to spam dictionary
            if len(words) > 0:
                spam_dictionary[spam_token] = words
            else:
                continue

        return spam_dictionary

    def retrieve_twin_word(self):
        spam_dictionary = dict()
        print("To be implemented")
        return spam_dictionary

    def retrieve_save(self, dict_type):
        spam_dictionary = dict()
        if dict_type is 0:
            # get only data muse
            print("getting only datamuse words")
            spam_dictionary = self.retrieve_data_muse()
        elif dict_type is 1:
            print("getting only twin words")
            # get only twin word
            spam_dictionary = self.retrieve_twin_word()
        else:
            print("getting from both")
            # get both

        # save the list
        self.save_list(spam_dictionary)

    def save_list(self, spam_dictionary):
        # save spam dictionary using pickle
        filename = self.dataset + "spam_dictionary.p"
        pickle.dump(spam_dictionary, open(filename, "wb"))
        print("saving spam dictionary")

    def get_list(self):
        # load the spam tokens
        filename = self.dataset + "spam_dictionary.p"
        spam_dictionary = pickle.load(open(filename, 'rb'))
        return spam_dictionary

    def get_words_per_spam_token(self, token):
        spam_dictionary = self.get_list()
        related_terms_list = spam_dictionary[token]
        return related_terms_list


if __name__ == '__main__':
    spam_dict = SpamDictionary()

    """ get online words and save them in a pickle file """
    # spam_dict.retrieve_save(0)

    """ get words per spam token """

    token = "free"
    spam_list = spam_dict.get_words_per_spam_token(token)
    print(spam_list)
    exit(0)
