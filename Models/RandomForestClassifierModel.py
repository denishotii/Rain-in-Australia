import pandas as pd
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, classification_report, confusion_matrix

data = pd.read_csv('weatherAUS_missing_values_inputed.csv')

Xs, Ys = data.drop(data.columns[-1], axis = 1), data[data.columns[-1]]

rf_params = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

X_train, X_test, Y_train, Y_test = train_test_split(Xs,
                                                    Ys,
                                                    train_size=0.2,
                                                    test_size=0.05,
                                                    shuffle=True,
                                                    random_state=42,
                                                    stratify=Ys)
t0 = time.time()

rf_model = RandomForestClassifier(**rf_params)

rf_model.fit(X_train, Y_train)

y_pred = rf_model.predict(X_test)

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

