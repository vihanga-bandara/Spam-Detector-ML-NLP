import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# Creating two fuzzy input variables and one output fuzzy variable
tweet_model = ctrl.Antecedent(np.arange(0, 101, 1), 'Tweet Model')
user_model = ctrl.Antecedent(np.arange(0, 101, 1), 'User Model')
predict_spam = ctrl.Consequent(np.arange(0, 100, 1), 'Spam Prediction')

# # Auto-membership function population is possible with .automf(3, 5, or 7)
# quality.automf(3)
# service.automf(3)

# using the automf to populate the fuzzy variables with terms
tweet_spam_names = ['not_spam', 'maybe_spam', 'definite_spam']
tweet_user_spam_names = ['not_spam', 'maybe_spam', 'definite_spam']
model_output_names = ['not_spam', 'spam']

tweet_model.automf(names=tweet_spam_names)
user_model.automf(names=tweet_user_spam_names)
user_model.automf(names=model_output_names)

# Custom membership functions can be built interactively with a familiar,
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# You can see how these look with .view()
tip['low'].view()

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

rule1.view()

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

# Crunch the numbers
tipping.compute()

print(tipping.output['tip'])
tip.view(sim=tipping)
