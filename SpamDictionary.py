import pickle
from datamuse import datamuse


class SpamDictionary:
    pickle = "pickle/"
    dataset = "dataset/"

    def __init__(self):
        print("Initialized Dictionary")

    def update_dictionary(self, tokens):

    def retrieve_data_muse(self):
        # load the spam tokens
        filename = self.dataset + "spam_tokens.p"
        spam_tokens = pickle.load(open(filename, 'rb'))

        # load the api
        api = datamuse.Datamuse()

        # words related to this word
        for spam_token in spam_tokens:
            datamuse_response_similar = api.words(ml=spam_token, max=5)
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

    def retrieve_twin_word(self):

    def retrieve_save(self, dict_type):
        words_list = []
        if dict_type is 0:
            # get only data muse
            print("getting only datamuse words")
            words_list = self.retrieve_data_muse()
        elif dict_type is 1:
            print("getting only twin words")
            # get only twin word
            words_list = self.retrieve_twin_word()
        else:
            print("getting from both")
            # get both

        # save the list
        self.save_list(words_list)

    def get_ml_muse(self):

    def get_rel_syn_muse(self)

    def get_assoc_twin(self):

    def save_list(self, words_list):

    def get_list(self):


if __name__ == '__main__':
    spam_dict = SpamDictionary()

    # get online words and save them in a pickle file
    spam_dict.retrieve_save(0)
