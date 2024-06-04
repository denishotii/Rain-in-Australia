import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('weatherAUS_missing_values_inputed.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Correlation Matrix
corr_matrix = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Correlation with 'RainTomorrow'
target = 'RainTomorrow'
corr_with_target = corr_matrix[target].sort_values(ascending=False)
print(corr_with_target)

# Define features and target variable
features = ['Rainfall', 'Humidity3pm', 'Humidity9am', 'Cloud3pm', 'Cloud9am', 'WindSpeed3pm', 'WindSpeed9am', 'Pressure3pm', 'Pressure9am', 'Temp3pm', 'Temp9am', 'MaxTemp', 'MinTemp']
target = 'RainTomorrow'

# Split the data into training and testing sets
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance - Random Forest')
plt.show()

# Create lagged features for time series prediction
def create_lagged_features(data, lag=1):
    df = data.copy()
    for i in range(1, lag+1):
        df[f'RainTomorrow_lag{i}'] = df['RainTomorrow'].shift(i)
    df.dropna(inplace=True)
    return df

lag = 7  # Predicting rain for the next 7 days
lagged_data = create_lagged_features(data, lag)

# Define features and target for lagged data
features_lagged = features + [f'RainTomorrow_lag{i}' for i in range(1, lag+1)]
target_lagged = 'RainTomorrow'

# Split the data
X_lagged = lagged_data[features_lagged]
y_lagged = lagged_data[target_lagged]

X_train_lagged, X_test_lagged, y_train_lagged, y_test_lagged = train_test_split(X_lagged, y_lagged, test_size=0.3, random_state=42)

# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train_lagged, y_train_lagged)

# Predict
y_pred_lagged = lr.predict(X_test_lagged)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test_lagged.values, label='Actual')
plt.plot(y_pred_lagged, label='Predicted')
plt.title('Actual vs Predicted Rain - Linear Regression')
plt.legend()
plt.show()

# Define flooding and drought thresholds
flood_threshold = data['Rainfall'].quantile(0.95)
drought_threshold = data['Rainfall'].quantile(0.05)

# Create target variables for flood and drought
data['Flood'] = (data['Rainfall'] > flood_threshold).astype(int)
data['Drought'] = (data['Rainfall'] < drought_threshold).astype(int)

# Feature importance for flood prediction
rf_flood = RandomForestClassifier(n_estimators=100, random_state=42)
rf_flood.fit(X_train, data.loc[X_train.index, 'Flood'])

feature_importance_flood = pd.Series(rf_flood.feature_importances_, index=features).sort_values(ascending=False)
print(feature_importance_flood)

# Plot feature importance for flood prediction
plt.figure(figsize=(10, 6))
feature_importance_flood.plot(kind='bar')
plt.title('Feature Importance for Flood Prediction')
plt.show()

# Feature importance for drought prediction
rf_drought = RandomForestClassifier(n_estimators=100, random_state=42)
rf_drought.fit(X_train, data.loc[X_train.index, 'Drought'])

feature_importance_drought = pd.Series(rf_drought.feature_importances_, index=features).sort_values(ascending=False)
print(feature_importance_drought)

# Plot feature importance for drought prediction
plt.figure(figsize=(10, 6))
feature_importance_drought.plot(kind='bar')
plt.title('Feature Importance for Drought Prediction')
plt.show()
