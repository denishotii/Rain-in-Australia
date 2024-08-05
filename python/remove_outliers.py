import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('weatherAUS.csv')


numeric_cols = data.select_dtypes(include = ['float64']).columns

for column in numeric_cols:

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    mean = data[column].median()

    IQR = Q3 - Q1

    upper = Q3 + IQR * 1.5
    lower = Q1 - IQR * 1.5

    data.loc[data[column] >= upper, column] = np.nan
    data.loc[data[column] <= lower, column] = np.nan



plt.figure(figsize=(25, 15))  

for column in range(len(numeric_cols)):
    plt.subplot(4, 4, column + 1)
    sns.boxplot(y = data[numeric_cols[column]])  
    plt.title(f'The boxplot of {numeric_cols[column]}')  
    
plt.tight_layout()
plt.show()