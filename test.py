from pandas_profiling import ProfileReport
import pandas as pd

data = pd.read_csv('weatherAUS.csv')

data = data.sample(frac = 0.1, random_state = 42)

profile = ProfileReport(data, title = 'Weather in Australia')

profile.to_notebook_iframe()

profile.to_file("report_2.html")