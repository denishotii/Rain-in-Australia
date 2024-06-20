import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv('weatherAUS_missing_values_inputed.csv')

# Separate features and target
Xs, Ys = data.drop(data.columns[-1], axis=1), data[data.columns[-1]]

# Random Forest parameters
rf_params = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

# Time series cross-validation setup
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

# Train the Random Forest model
t0 = time.time()
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, Y_train)
y_pred = rf_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, rf_model.predict_proba(X_test)[:, 1])  # Use probabilities for roc_auc_score
coh_kap = cohen_kappa_score(Y_test, y_pred)
time_taken = time.time() - t0

# Print metrics
print(f"Accuracy = {accuracy}")
print(f"ROC Area under Curve = {roc_auc}")
print(f"Cohen's Kappa = {coh_kap}")
print(f"Time taken = {time_taken}")
print(classification_report(Y_test, y_pred, digits=5))

# Plot confusion matrix
cf_matrix = confusion_matrix(Y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
