import optuna
import pandas as pd

parameters = ["salt", "coriander", "cumin", "turmeric", "garam_masala", "yogurt"]

parameters_min, parameters_max = {}, {}
parameters_min["salt"], parameters_max["salt"] = 2.0, 6.0
parameters_min["coriander"], parameters_max["coriander"] = 2.0, 6.0
parameters_min["cumin"], parameters_max["cumin"] = 0.0, 4.0
parameters_min["turmeric"], parameters_max["turmeric"] = 0.0, 2.0
parameters_min["garam_masala"], parameters_max["garam_masala"] = 0.0, 4.0
parameters_min["yogurt"], parameters_max["yogurt"] = 0, 60

search_space = {
    "salt": optuna.distributions.FloatDistribution(parameters_min["salt"], parameters_max["salt"]),
    "coriander": optuna.distributions.FloatDistribution(parameters_min["coriander"], parameters_max["coriander"]),
    "cumin": optuna.distributions.FloatDistribution(parameters_min["cumin"], parameters_max["cumin"]),
    "turmeric": optuna.distributions.FloatDistribution(parameters_min["turmeric"], parameters_max["turmeric"]),
    "garam_masala": optuna.distributions.FloatDistribution(parameters_min["garam_masala"], parameters_max["garam_masala"]),
    "yogurt": optuna.distributions.FloatDistribution(parameters_min["yogurt"], parameters_max["yogurt"]),
}
score_column = "score"

# load suggested parameters from data.csv
name = "data.csv"
data = pd.read_csv(name)

# create study object
study = optuna.create_study(direction="maximize")

# register past trials
for i, row in data.iterrows():
    params = {}
    for parameter in parameters:
        params[parameter] = row[parameter]
    trial = optuna.trial.create_trial(
        params=params,
        distributions=search_space,
        value=row[score_column],
    )
    study.add_trial(trial)

# suggest new parameters
trial = study.ask(search_space)
rounded_trial = {parameter: round(trial.params[parameter], 1) for parameter in parameters}
print("Trial parameters:")
for parameter in parameters:
    print(f"{parameter}: {rounded_trial[parameter]}")

# save suggested parameters to data.csv
data = data._append(rounded_trial, ignore_index=True)
data[score_column] = 0
data.to_csv(name, index=False)