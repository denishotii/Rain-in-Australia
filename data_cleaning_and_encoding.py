import pandas as pd

data = pd.read_csv('weatherAUS.csv')



"""
Dropping rows which target values are missing.
Before drop: 145,360 instances
After drop: 140,787  instances

Total loss: 4,573 instances - 3.15% of dataset
"""

data.dropna(subset = data.columns[-1], inplace = True)
data.dropna(subset = data.columns[-2], inplace = True)



"""
Encoding 'Yes' and 'No' values with 1.0 and 0.0 for the sake of regression models.
As it seems only 'RainToday' and 'RainTomorrow' has replaced here.
"""

data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0}).astype(float)
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype(float)



"""
Encoding 'Location', 'WindGustDir', 'WindDir9am', 'WindDir9am' columns with Frequency Encoding and normalizing them.
"""

city_counts = data['Location'].value_counts(normalize = True)
data['Location'] = data['Location'].map(city_counts)

wind_gust_dir_counts = data['WindGustDir'].value_counts(normalize = True)
data['WindGustDir'] = data['WindGustDir'].map(wind_gust_dir_counts)

wind_dir_9am_counts = data['WindDir9am'].value_counts(normalize = True)
data['WindDir9am'] = data['WindDir9am'].map(wind_dir_9am_counts)

wind_dir_3pm_counts = data['WindDir3pm'].value_counts(normalize = True)
data['WindDir9am'] = data['WindDir3pm'].map(wind_dir_3pm_counts)

new_names = {'Location': 'City_Frequency',
             'WindGustDir': 'WindGustDir_Frequency',
             'WindDir9am': 'WindDir9am_Frequency',
             'WindDir9am': 'WindDir9am_Frequency'
             }

data.rename(new_names, inplace = True)



"""
Exporting cleaned and encoded data into a new file in order to get rid of extra computation in the beginning of each process.
We do not overwrite for versioning purposes.
"""

updated_filepath = 'weatherAUS_cleaned_encoded.csv'

data.to_csv(updated_filepath, index = False)