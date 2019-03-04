import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

data = pd.read_csv('training_data_2_csv_UTF.csv')
SpamUsers = data[data.bot == 1]
NonSpamUsers = data[data.bot == 1]

bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget|expos|kill|bbb|truthe|fake|anony|free|virus|funky|RNA|jargon|nerd|swag|jack|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb|ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

data['screen_name_binary'] = data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
data['name_binary'] = data.name.str.contains(bag_of_words_bot, case=False, na=False)
data['description_binary'] = data.description.str.contains(bag_of_words_bot, case=False, na=False)
data['status_binary'] = data.status.str.contains(bag_of_words_bot, case=False, na=False)

data['listed_count_binary'] = (data.listed_count > 20000) == False

features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count',
            'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split

X = data[features].iloc[:, :-1]
y = data[features].iloc[:, -1]

clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print("Training Accuracy: %.5f" % accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" % accuracy_score(y_test, y_pred_test))

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
