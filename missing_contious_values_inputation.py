import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('weatherAUS_cleaned_encoded.csv')

dates = pd.to_datetime(data['Date'])

data.drop('Date', axis = 1, inplace = True)

data.insert(0, 'Month', dates.dt.month.astype(float))
data.insert(0, 'Day', dates.dt.day.astype(float))

"""
Columns which contaion continuous data and null values.
"""

null_contained_cols = ['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

for target in null_contained_cols:
    
    """
    Dropping columns which target contains NaN values and saving corresponding features in X_target.
    It will be used for predicting dropped NaN values.
    """
    indices = data.index[data[target].isna()]
    X_target = data.loc[indices].drop(target, axis = 1)
    
    dropped_data = data.dropna(subset = target)
    
    dropped_data = dropped_data.sample(frac = 1, random_state = 42)
    
    """
    As there is no test case, we train our model with full dataset (after dropping NaN target values).
    """
    X_train, Y_train = dropped_data.drop(target, axis = 1), dropped_data[target]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    train_set = lgb.Dataset(X_train, label=Y_train)
    
    """
    Parameters of LightGBM model
    """
    
    params = {'objective': 'regression',    # using regression model
              'boosting': 'gbdt',           # model uses Gradient Boosting which aimed to minimize loss (error)
              'learning_rate': 0.1,         # the step which model makes in each iteration for minimizing loss
              'num_leaves': 100,            # number of leaves for tree model
              'verbose': -1,                # force model not to show additional text information
              'device_type': 'cpu'}         # it says to algorithm to use cpu which is faster than gpu in this case


    model = lgb.train(params, train_set)
    
    Y_target = model.predict(X_target, num_iteration = model.best_iteration)
    
    """
    Writing predicted targets back to the data which also helps to have richer dataset in each iteration.
    """
    data.loc[indices, target] = Y_target

data.to_csv('weatherAUS_missing_continous_values_inputed.csv')