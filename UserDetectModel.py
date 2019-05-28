import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
import pickle


class UserDetectModel:
    def __init__(self):
        print("Initialise User Detect Model")

    def load_dataset(self):
        # importing spam user training dataset
        data = pd.read_csv('dataset/training_data_2_csv_UTF.csv')

        # breaking the dataset into spam and ham
        SpamUsers = data[data.bot == 1]
        NonSpamUsers = data[data.bot == 0]

        SpamUsers.info()
        NonSpamUsers.info()

        return data

    def preprocess(data):
        # basic bag of words model
        # Maybe import the bag of words from a file that contins the words - this allows updating the model
        bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|suspend|yes_ofc|forget|expos|kill|bbb|truthe|fake|anony|free|virus|funky|RNA|jargon|nerd|swag|jack|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb|ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

        # Feature Engineering (some more relationships to be added)

        # check the screen name for words in the BoW
        data['screen_name_binary'] = data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)

        # check the name for words in the BoW
        data['name_binary'] = data.name.str.contains(bag_of_words_bot, case=False, na=False)

        # check the description for words in the BoW
        data['description_binary'] = data.description.str.contains(bag_of_words_bot, case=False, na=False)

        # check the sstatus for words in the BoW
        data['status_binary'] = data.status.str.contains(bag_of_words_bot, case=False, na=False)

        # check the number of public lists that the user is a part of
        data['listed_count_binary'] = (data.listed_count > 20000) == False

        # Finalizing the feature set
        features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified',
                    'followers_count',
                    'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

        return features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        """Grid Search and Cross Validation to identify best parameters"""

        """ Decision Tree Classifier """
        # dt_classifier = DecisionTreeClassifier()
        # params_dt = {"min_samples_leaf": [5, 50, 100],
        #              "max_depth": [5, 6, 7, 8, 9, 10],
        #              "criterion": ['gini', 'entropy']
        #              }
        #
        # # use gridsearch to test all values
        # dt_gs = GridSearchCV(dt_classifier, params_dt, cv=5, n_jobs=2)
        # # fit model to training data
        # dt_gs.fit(features_set_train, y_train)
        #
        # # save best model
        # dt_best = dt_gs.best_estimator_
        # # check best n_estimators value
        # print(dt_gs.best_params_)

        """ END OF GRID SEARCH - best params have been used for model """

        clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)

        DecisionTreeClf = clf.fit(X_train, y_train)

        # predict the training dataset
        y_pred_train = DecisionTreeClf.predict(X_train)

        # predict the test dataset
        y_pred_test = DecisionTreeClf.predict(X_test)

        # score of the test dataset
        y_score_test = DecisionTreeClf.score(X_test, y_test)

        # Output classifier results
        print("Training Accuracy: %.5f" % accuracy_score(y_train, y_pred_train))
        print("Test Accuracy: %.5f" % accuracy_score(y_test, y_pred_test))

        return clf, X_train, X_test, y_train, y_test, y_pred_test

    def save_model(self, model):
        # save model using pickle
        filename = 'SpamUserDetectModel.sav'
        pickle.dump(model, open(filename, 'wb'))

    def performance_eval(self, clf, X_train, X_test, y_train, y_test, y_pred_test):
        # confusion_matrix = pd.DataFrame(
        #     confusion_matrix(y_test, y_pred_test),
        #     index=[['actual', 'actual '], ['ham', 'spam']],
        #     columns=[['predicted', 'predicted'], ['ham', 'spam']])

        print(confusion_matrix)

        cm = confusion_matrix

        # Visualizing the Training set results
        from matplotlib.colors import ListedColormap

        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

        plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)

        plt.title('Decision Tree Classifier (Training Set)')
        plt.legend()
        plt.show()

        ax = plt.subplot()
        svm = sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Greens')
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['ham', 'spam']);
        ax.yaxis.set_ticklabels(['ham', 'spam'])
        figure = svm.get_figure()
        figure.savefig('images/confusion_matrix_user.png', dpi=400)
        plt.close()
        # predict probabilities
        probs = clf.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.3f' % auc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        # plot no skill
        pyplot.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        pyplot.plot(fpr, tpr, marker='.')
        # show the plot
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')

        pyplot.savefig('images/roc_curve_user.png', dpi=400)
        pyplot.show()
        pyplot.close()
        print(confusion_matrix)

        print(classification_report(y_test, y_pred_test))

        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        # calculate F1 score
        f1 = f1_score(y_test, y_pred_test)
        # calculate precision-recall AUC
        auc = auc(recall, precision)
        # calculate average precision score
        ap = average_precision_score(y_test, probs)
        print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
        # plot no skill
        pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
        # plot the precision-recall curve for the model
        pyplot.plot(recall, precision, marker='.')
        # show the plot
        pyplot.savefig('images/pr_curve.png', dpi=400)
        pyplot.show()
        pyplot.close()

    def main(self):
        # load the dataset
        dataset = self.load_dataset()

        # get the required feature engineered variables from the dataset by preprocess
        features_set = self.preprocess(dataset)

        # using the feature set arrange the dataset
        X = dataset[features_set].iloc[:, :-1]
        y = dataset[features_set].iloc[:, -1]

        # train the model using the features identified
        clf, X_train, X_test, y_train, y_test, y_pred_test = self.train_model(X, y)

        # save the model to pickle
        self.save_model(clf)

        # run performance graphs
        self.performance_eval(clf, X_train, X_test, y_train, y_test, y_pred_test)
