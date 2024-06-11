import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


data = pd.read_csv('weatherAUS_missing_values_inputed.csv')

Xs, Ys = data.drop(data.columns[-1], axis = 1), data[data.columns[-1]]

ts_cv = TimeSeriesSplit(
    n_splits=3,
    gap=48,
    max_train_size=10000,
    test_size=3000,
)

all_splits = list(ts_cv.split(Xs, Ys))
train_idx, test_idx = all_splits[0]

X_train, X_test = Xs.iloc[train_idx, :], Xs.iloc[test_idx, :]
Y_train, Y_test = Ys.iloc[train_idx], Ys.iloc[test_idx]

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l1', solver='saga', max_iter=10_000)
)

#param_C = [i for i in  np.linspace(3, 4, 101)]

grid_search_cv = GridSearchCV(
    pipeline,
    param_grid = {'logisticregression__C': [3]},
    cv = 5,
    n_jobs = 8,
    scoring='f1'
)

grid_search_cv.fit(X_train, Y_train)


best_k = grid_search_cv.best_params_['logisticregression__C']
best_model = grid_search_cv.best_estimator_

score = best_model.score(X_test, Y_test)

print('Best k: ', best_k)
print('Test Score: ', score)

"""
Best K is 3.08 for train_size = 0.2 and test_size = 0.05
"""

y_proba = best_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(Y_test, y_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()