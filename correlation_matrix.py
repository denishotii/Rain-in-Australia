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

# Display data types of each column
print(data.dtypes)

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Filter numeric columns
numeric_data = data.select_dtypes(include=[float, int])

# Compute the correlation matrix
corr_matrix = numeric_data.corr()

# Display the correlation matrix
print(corr_matrix)

# Plot the heatmap for correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')  # Limiting to two decimal places
plt.title('Correlation Matrix')
plt.show()

# Correlation with target variable 'RainTomorrow'
target = 'RainTomorrow'
corr_with_target = corr_matrix[target].sort_values(ascending=False)
print(corr_with_target)
