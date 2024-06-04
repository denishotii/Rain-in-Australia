"""
k_neighbors = np.arange(10, 25)

pipeline = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler()),
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



grid_search.fit(pd.DataFrame(X_train), Y_train)

end = time.time()

print(grid_search.best_params_['knn__n_neighbors'])
print(grid_search.best_score_)
print(end - start)
print(grid_search.score(pd.DataFrame(X_test), Y_test))
"""

import pandas as pd

data = pd.read_csv('weatherAUS_missing_values_inputed.csv')

print(data.info())