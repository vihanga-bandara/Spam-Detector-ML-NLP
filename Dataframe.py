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
        exists = os.path.isfile('./data/rt.p')
        if exists:
            filename = self.data + "rt.p"
            rt = pickle.load(open(filename, 'rb'))
            l = len(rt)
            check = [item for item in rt if item not in t]
            return len(check) is l - 1 and 1 or 0

        return
