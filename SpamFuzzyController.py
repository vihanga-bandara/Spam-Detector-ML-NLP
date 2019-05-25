import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# # Creating two fuzzy input variables and one output fuzzy variable
# tweet_model = ctrl.Antecedent(np.arange(0, 101, 1), 'Tweet_Model')
# user_model = ctrl.Antecedent(np.arange(0, 101, 1), 'User_Model')
# predict_spam = ctrl.Consequent(np.arange(0, 100, 1), 'Spam_Prediction')
#
# # Creating memberships for tweet model - Input Variable
# tweet_model['not_spam'] = fuzz.trimf(tweet_model.universe, [0, 0, 51])
# tweet_model['maybe_spam'] = fuzz.trimf(tweet_model.universe, [0, 51, 101])
# tweet_model['definite_spam'] = fuzz.trimf(tweet_model.universe, [51, 101, 101])
#
# # Creating memberships for user model - Input Variable
# user_model['not_spam'] = fuzz.trimf(user_model.universe, [0, 0, 51])
# user_model['maybe_spam'] = fuzz.trimf(user_model.universe, [0, 51, 101])
# user_model['definite_spam'] = fuzz.trimf(user_model.universe, [51, 101, 101])
#
# # Creating memberships for tweet model - Output Variable
# predict_spam['not_spam'] = fuzz.trimf(predict_spam.universe, [0, 0, 101])
# predict_spam['spam'] = fuzz.trimf(predict_spam.universe, [0, 101, 101])
#
# # view a graph showing the memberships of the variables initialized
# tweet_model.view()
# user_model.view()
# predict_spam.view()
#
# # initiating rules for the spam fuzzy controller 3^2 probability therefore nine rules will be applied
# rule1 = ctrl.Rule(tweet_model['not_spam'] & user_model['not_spam'], predict_spam['not_spam'])
# rule2 = ctrl.Rule(tweet_model['not_spam'] & user_model['maybe_spam'], predict_spam['not_spam'])
# rule3 = ctrl.Rule(tweet_model['not_spam'] & user_model['definite_spam'], predict_spam['spam'])
#
# rule4 = ctrl.Rule(tweet_model['maybe_spam'] & user_model['not_spam'], predict_spam['not_spam'])
# rule5 = ctrl.Rule(tweet_model['maybe_spam'] & user_model['maybe_spam'], predict_spam['spam'])
# rule6 = ctrl.Rule(tweet_model['maybe_spam'] & user_model['definite_spam'], predict_spam['spam'])
#
# rule7 = ctrl.Rule(tweet_model['definite_spam'] & user_model['not_spam'], predict_spam['spam'])
# rule8 = ctrl.Rule(tweet_model['definite_spam'] & user_model['maybe_spam'], predict_spam['spam'])
# rule9 = ctrl.Rule(tweet_model['definite_spam'] & user_model['definite_spam'], predict_spam['spam'])
#
# # Add the rules to a new ControlSystem
# predict_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
# spam_percentage = ctrl.ControlSystemSimulation(predict_control)
#
# spam_percentage.input['Tweet_Model'] = 50
# spam_percentage.input['User_Model'] = 50
#
# # Crunch the numbers
# spam_percentage.compute()
#
# print(spam_percentage.output['Spam_Prediction'])
# predict_spam.view(sim=spam_percentage)


class SpamFuzzyController:
    _spam_percentage = None

    def __init__(self):
        print('Initializing Fuzzy Logic')

    def fuzzy_initialize(self):
        # Creating two fuzzy input variables and one output fuzzy variable
        tweet_model = ctrl.Antecedent(np.arange(0, 101, 1), 'Tweet_Model')
        user_model = ctrl.Antecedent(np.arange(0, 101, 1), 'User_Model')
        predict_spam = ctrl.Consequent(np.arange(0, 100, 1), 'Spam_Prediction')

        # Creating memberships for tweet model - Input Variable
        tweet_model['not_spam'] = fuzz.trimf(tweet_model.universe, [0, 0, 49])
        tweet_model['maybe_spam'] = fuzz.trimf(tweet_model.universe, [50, 50, 101])
        tweet_model['definite_spam'] = fuzz.trimf(tweet_model.universe, [50, 101, 101])

        # Creating memberships for user model - Input Variable
        user_model['not_spam'] = fuzz.trimf(user_model.universe, [0, 0, 51])
        user_model['maybe_spam'] = fuzz.trimf(user_model.universe, [0, 51, 101])
        user_model['definite_spam'] = fuzz.trimf(user_model.universe, [51, 101, 101])

        # Creating memberships for tweet model - Output Variable
        predict_spam['not_spam'] = fuzz.trimf(predict_spam.universe, [0, 0, 51])
        predict_spam['maybe_spam'] = fuzz.trimf(predict_spam.universe, [0, 51, 101])
        predict_spam['spam'] = fuzz.trimf(predict_spam.universe, [51, 101, 101])

        # # view a graph showing the memberships of the variables initialized
        # tweet_model.view()
        # user_model.view()
        # predict_spam.view()

        # initiating rules for the spam fuzzy controller 3^2 probability therefore nine rules will be applied
        rule1 = ctrl.Rule(tweet_model['not_spam'] & user_model['not_spam'], predict_spam['not_spam'])
        rule2 = ctrl.Rule(tweet_model['not_spam'] & user_model['maybe_spam'], predict_spam['not_spam'])
        rule3 = ctrl.Rule(tweet_model['not_spam'] & user_model['definite_spam'], predict_spam['maybe_spam'])

        rule4 = ctrl.Rule(tweet_model['maybe_spam'] & user_model['not_spam'], predict_spam['not_spam'])
        rule5 = ctrl.Rule(tweet_model['maybe_spam'] & user_model['maybe_spam'], predict_spam['spam'])
        rule6 = ctrl.Rule(tweet_model['maybe_spam'] & user_model['definite_spam'], predict_spam['spam'])

        rule7 = ctrl.Rule(tweet_model['definite_spam'] & user_model['not_spam'], predict_spam['maybe_spam'])
        rule8 = ctrl.Rule(tweet_model['definite_spam'] & user_model['maybe_spam'], predict_spam['spam'])
        rule9 = ctrl.Rule(tweet_model['definite_spam'] & user_model['definite_spam'], predict_spam['spam'])

        # Add the rules to a new ControlSystem
        predict_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self._spam_percentage = ctrl.ControlSystemSimulation(predict_control)

    def fuzzy_predict(self, tweet_proba, user_model_proba):
        spam_percentage = self._spam_percentage
        spam_percentage.input['Tweet_Model'] = tweet_proba
        spam_percentage.input['User_Model'] = user_model_proba

        # Crunch the numbers
        spam_percentage.compute()

        print(" Tweet Score - {0} | User Score = {1} | Fuzzy Prediction = {2}".format(tweet_proba, user_model_proba,
                                                                                      spam_percentage.output[
                                                                                          'Spam_Prediction']))
        final_spam_score = spam_percentage.output['Spam_Prediction']
        return int(round(final_spam_score))


if __name__ == '__main__':
    spamfuz = SpamFuzzyController()
    spamfuz.fuzzy_initialize()
    spam_score_fuzzy = spamfuz.fuzzy_predict(40, 40)
    spam_score_fuzzy = spamfuz.fuzzy_predict(50, 50)
    spam_score_fuzzy = spamfuz.fuzzy_predict(10, 10)
    spam_score_fuzzy = spamfuz.fuzzy_predict(40, 60)
    spam_score_fuzzy = spamfuz.fuzzy_predict(60, 60)
    spam_score_fuzzy = spamfuz.fuzzy_predict(80, 80)
    spam_score_fuzzy = spamfuz.fuzzy_predict(90, 90)
    spam_score_fuzzy = spamfuz.fuzzy_predict(20, 90)
    spam_score_fuzzy = spamfuz.fuzzy_predict(40, 90)
    spam_score_fuzzy = spamfuz.fuzzy_predict(90, 20)
    spam_score_fuzzy = spamfuz.fuzzy_predict(90, 40)
    spam_score_fuzzy = spamfuz.fuzzy_predict(45, 70)
    spam_score_fuzzy = spamfuz.fuzzy_predict(70, 60)
    spam_score_fuzzy = spamfuz.fuzzy_predict(51, 90)
    exit(0)
