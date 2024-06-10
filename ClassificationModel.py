import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, f1_score

data = pd.read_csv('weatherAUS_missing_values_inputed.csv')

Xs, Ys = data.drop(data.columns[-1], axis = 1), data[data.columns[-1]]

X_train, X_test, Y_train, Y_test = train_test_split(Xs,
                                                    Ys,
                                                    train_size=0.2,
                                                    test_size=0.05,
                                                    shuffle=True,
                                                    random_state=42,
                                                    stratify=Ys)

pipeline = Pipeline([
    ('knn', KNeighborsClassifier())
])


n_neighbors = np.arange(1, 20)

grid_search_cv = GridSearchCV(
    pipeline,
    param_grid = {'knn__n_neighbors': n_neighbors},
    cv=5,
    scoring='f1',
    n_jobs=8
)

grid_search_cv.fit(X_train, Y_train)

best_k = grid_search_cv.best_params_['knn__n_neighbors']
best_model = grid_search_cv.best_estimator_

score = best_model.score(X_test, Y_test)

print('Best k: ', best_k)
print('Test Score: ', score)


y_proba = cross_val_predict(best_model, X_train, Y_train, cv = 5, method='predict_proba')[:, 1]

precision, recall, thresholds = precision_recall_curve(Y_train, y_proba)

print(precision)
print(recall)
print(thresholds)

scores = []

for i in range(len(precision)):
    score = (2*precision[i]*recall[i]) / (precision[i] + recall[i])
    scores.append(score)
    print(score)

print(thresholds[np.argmax(scores)])