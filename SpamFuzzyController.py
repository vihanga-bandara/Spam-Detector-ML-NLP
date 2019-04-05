import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Creating two fuzzy input variables and one output fuzzy variable
tweet_model = ctrl.Antecedent(np.arange(0, 101, 1), 'Tweet Model')
user_model = ctrl.Antecedent(np.arange(0, 101, 1), 'User Model')
predict_spam = ctrl.Consequent(np.arange(0, 100, 1), 'Spam Prediction')

# Creating memberships for tweet model - Input Variable
tweet_model['not_spam'] = fuzz.trimf(tweet_model.universe, [0, 0, 51])
tweet_model['maybe_spam'] = fuzz.trimf(tweet_model.universe, [0, 51, 101])
tweet_model['definite_spam'] = fuzz.trimf(tweet_model.universe, [51, 101, 101])

# Creating memberships for user model - Input Variable
user_model['not_spam'] = fuzz.trimf(user_model.universe, [0, 0, 51])
user_model['maybe_spam'] = fuzz.trimf(user_model.universe, [0, 51, 101])
user_model['definite_spam'] = fuzz.trimf(user_model.universe, [51, 101, 101])

# Creating memberships for tweet model - Output Variable
predict_spam['not_spam'] = fuzz.trimf(predict_spam.universe, [-51, 0, 51])
predict_spam['spam'] = fuzz.trimf(predict_spam.universe, [51, 101, 101])

# view a graph showing the memberships of the variables initialized
tweet_model.view()
user_model.view()
predict_spam.view()

# initiating rules for the spam fuzzy controller 3^2 probability therefore nine rules will be applied
rule1 = ctrl.Rule(tweet_model['not_spam'] | user_model['not_spam'], predict_spam['not_spam'])
rule2 = ctrl.Rule(tweet_model['not_spam'] | user_model['maybe_spam'], predict_spam['not_spam'])
rule3 = ctrl.Rule(tweet_model['not_spam'] | user_model['definite_spam'], predict_spam['spam'])

rule4 = ctrl.Rule(tweet_model['maybe_spam'] | user_model['not_spam'], predict_spam['not_spam'])
rule5 = ctrl.Rule(tweet_model['maybe_spam'] | user_model['maybe_spam'], predict_spam['spam'])
rule6 = ctrl.Rule(tweet_model['maybe_spam'] | user_model['definite_spam'], predict_spam['spam'])

rule7 = ctrl.Rule(tweet_model['definite_spam'] | user_model['not_spam'], predict_spam['spam'])
rule8 = ctrl.Rule(tweet_model['definite_spam'] | user_model['maybe_spam'], predict_spam['spam'])
rule9 = ctrl.Rule(tweet_model['definite_spam'] | user_model['definite_spam'], predict_spam['spam'])

# Add the rules to a new ControlSystem
predict_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
spam_percentage = ctrl.ControlSystemSimulation(predict_control)

spam_percentage.input['quality'] = 6.5
spam_percentage.input['service'] = 9.8

# Crunch the numbers
spam_percentage.compute()

print(spam_percentage.output['predict_spam'])
predict_spam.view(sim=spam_percentage)
