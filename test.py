import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time


data = pd.read_csv('weatherAUS_cleaned_encoded.csv')

print(data)

"""

Xs, Ys = data.drop(data.columns[-1], axis = 1), data[data.columns[-1]]

Ys_small_test = Ys.iloc[: int((0.5 * len(Ys)))]

Xs_small_test = Xs.iloc[: int((0.5 * len(Xs)))]

X_train, X_test, Y_train, Y_test = train_test_split(Xs_small_test,
                                                    Ys_small_test,
                                                    train_size = 0.8,
                                                    shuffle = True)


#print(X_train_loc)

k_neighbors = np.arange(1, 31)

pipeline = Pipeline(steps = [
    #('imputer', SimpleImputer(strategy = 'mean')),
    #('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
    ]
)

start = time.time()

grid_search = GridSearchCV(
    pipeline,
    param_grid = {'knn__n_neighbors': k_neighbors},
    cv = 5,
    scoring = 'accuracy',
    return_train_score = True,
    n_jobs=8)



grid_search.fit(pd.DataFrame(X_train['Location']), Y_train)

end = time.time()

print(grid_search.best_params_['knn__n_neighbors'])
print(grid_search.best_score_)
print(end - start)
print(grid_search.score(pd.DataFrame(X_test['Location']), Y_test))

"""