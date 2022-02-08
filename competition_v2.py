# %% markdown
# # Header
# - bullet point
#

#%% 
# Libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport 
import missingno as msno
import matplotlib.pyplot as plt

# Prophet dependencies
from prophet import Prophet
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#plotly.offline.init_notebook_mode(connected=True)
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot



# %% Import
store = pd.read_csv("./data/store.csv")
sales = pd.read_csv("./data/train.csv")

# drop customer col
sales = sales.drop('Customers', axis=1)
#sales = sales.drop(columns='Customers')

# drop NA's from sales (2.7% only)
sales = sales.dropna()

# check NA matrix
%matplotlib inline
msno.matrix(store)

# profiler
#profile = ProfileReport(store, title="Pandas Profiling Report", explorative=True)
#profile

# convert cols to int
to_int = ["Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday"]
for i in to_int:
    sales[i] = sales[i].astype(np.int64)

# convert StateHoliday to Cat
sales['StateHoliday'] = pd.Categorical(sales['StateHoliday'])
sales['Date'] = pd.to_datetime(sales["Date"])


# run model on single shop
single_sales = sales.query("Store == 1")
single_sales = single_sales.drop(["StateHoliday", "SchoolHoliday"], axis=1)

#%% Baseline model with mean




#%% Prophet

# prophet
single_sales.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

# split last 10 days from shop==1
X_train = single_sales[single_sales.ds<='2014-07-20'][['ds', 'y']]
X_test = single_sales[single_sales.ds>'2014-07-20'][['ds']]

# fit
model = Prophet()
model.fit(X_train)

# check the predictions for the training data
pred_train = model.predict(X_train)

# use the trained model to make a forecast
pred_test = model.predict(X_test)

p = model.plot(pred_test)