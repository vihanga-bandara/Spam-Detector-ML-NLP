import os
import pickle


class Dataframe:
    data = "data/"

    def __init__(self):
        print("initiated")

    def check_availability(self, t):
        exists = os.path.isfile('./data/rt.p')
        if exists:
            filename = self.data + "rt.p"
            rt = pickle.load(open(filename, 'rb'))
            rt.extend(t)
            filename = self.data + "rt.p"
            pickle.dump(rt, open(filename, "wb"))
        else:
            filename = self.data + "rt.p"
            pickle.dump(t, open(filename, "wb"))

    def find(self, t):
        filename = self.data + "rt.p"
        rt = pickle.load(open(filename, 'rb'))
        check = [item for item in t if item not in rt]
        return len(check) is 0 and 10 or 11
