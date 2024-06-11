import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, classification_report, confusion_matrix

data = pd.read_csv('weatherAUS_missing_values_inputed.csv')

Xs, Ys = data.drop(data.columns[-1], axis = 1), data[data.columns[-1]]

xgb_params ={'n_estimators': 500,
            'max_depth': 16}

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
t0 = time.time()

xgb_model = xgb.XGBClassifier(**xgb_params)


xgb_model.fit(X_train, Y_train)

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, y_pred) 
coh_kap = cohen_kappa_score(Y_test, y_pred)
time_taken = time.time() - t0

print(f"Accuracy = {accuracy}")
print(f"ROC Area under Curve = {roc_auc}")
print(f"Cohen's Kappa = {coh_kap}")
print(f"Time taken = {time_taken}")
print(classification_report(Y_test,y_pred,digits=5))

cf_matrix = confusion_matrix(Y_test, y_pred)
norm = plt.Normalize(-100,100)
sns.heatmap(cf_matrix, annot=True)