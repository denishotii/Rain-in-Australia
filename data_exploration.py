import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('weatherAUS.csv')

"""
Dropping rows which target values are missing.
Before drop: 145,360 instances
After drop: 140,787  instances

Total loss: 4,573 instances - 3.15% of dataset
"""

data.dropna(subset = data.columns[-1], inplace = True)
data.dropna(subset = data.columns[-2], inplace = True)



"""
Encoding 'Yes' and 'No' values with 1.0 and 0.0 for the sake of regression models.
As it seems only 'RainToday' and 'RainTomorrow' has replaced here.
"""

data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0}).astype(float)
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype(float)


Xs, Ys = data.drop(data.columns[-1], axis = 1), data[data.columns[-1]]

Ys_small_test = Ys.iloc[: int((0.2 * len(Ys)))]

Xs_small_test = Xs.iloc[: int((0.2 * len(Xs)))]

X_train, X_test, Y_train, Y_test = train_test_split(Xs_small_test,
                                                    Ys_small_test,
                                                    train_size = 0.8,
                                                    shuffle = False)

k_neighbors = np.arange(1, 31)

pipeline = Pipeline(steps = [
    #('imputer', SimpleImputer(strategy = 'mean')),
    #('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
    ]
)

grid_search = GridSearchCV(
    pipeline,
    param_grid = {'knn__n_neighbors': k_neighbors},
    cv = 5,
    return_train_score = True)

grid_search.fit(X_train['Location'], Y_train)

print(grid_search.best_params_['knn__n_neighbors'])