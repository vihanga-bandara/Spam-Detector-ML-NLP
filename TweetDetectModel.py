import sys
import sklearn
import pandas as pd
import numpy as np
import nltk
import matplotlib


class TweetDetectModel:
    tweet_dataset = None

    def get_package_versions(self):
        package_version = dict()
        package_version["Python"] = 'Python: {}', format(sys.version)
        package_version["Python"] = 'NLTK: {}', format(nltk.__version__)
        package_version["Python"] = 'Scikit-learn: {}', format(sklearn.__version__)
        package_version["Python"] = 'Pandas: {}', format(pd.__version__)
        package_version["Python"] = 'Numpy: {}', format(np.__version__)
        package_version["Python"] = 'MatPlotLib: {}', format(matplotlib.__version__)
        return package_version

    def load_dataset(self):
        # try to load the dataset
        try:
            df = pd.read_csv('dataset/SpamTweetsFinalDataset.csv', header=None)
            print(df.info())
            print(df.head())

            # check the class balance ratio / distribution
            column_names = list(df.head(0))
            classes = df[column_names[1]].str.strip()
            print(classes.value_counts())
            self.tweet_dataset = df
            return True

        except Exception as e:
            print("Error loading the dataset")
            return False

    def main(self):
        package_version = self.__get_package_versions()
        if self.load_dataset():
