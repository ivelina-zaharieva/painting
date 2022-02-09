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
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder

# # Prophet dependencies
# from prophet import Prophet
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# plotly.offline.init_notebook_mode(connected=True)
# from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot


# %% Import
store = pd.read_csv("./data/store.csv")
sales = pd.read_csv("./data/train.csv")

# %% Prep data
# drop customer col
#sales = sales.drop('Customers', axis=1)

# drop NA's from sales (2.7% only)
sales = sales.dropna()
# profiler
#profile = ProfileReport(store, title="Pandas Profiling Report", explorative=True)
#profile

# convert cols to categorical
to_cat = ["Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "StateHoliday"]
for i in to_cat:
    sales.loc[:, i] = pd.Categorical(sales[i])

store["Store"] = pd.Categorical(store["Store"])

# convert Date to DateTime
sales['Date'] = pd.to_datetime(sales["Date"])

# %% Merge tables

rossman_df = pd.merge(sales, store, how='left', on='Store')


# %% sanity check
rossman_df = rossman_df[~((rossman_df.Sales<1)&(rossman_df.Open==1))]


# %% Extract date, time features

def add_time_features(df):
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['WeekOfYear'] = df.Date.dt.weekofyear
    return df

# %% Target encode function

def encode_target(column, target):
    encoder = TargetEncoder()
    encoded_column = encoder.fit_transform(column, target)
    return encoded_column

# %% Transform columns in rossman_df

rossman_df_enc = add_time_features(rossman_df)

for i in to_cat:
    rossman_df_enc.loc[:, i] = encode_target(rossman_df_enc.loc[:, i], rossman_df_enc["Sales"])


# %%
encoder = TargetEncoder()
rossman_df_enc.loc[:, "Store"] = encoder.fit_transform(
    rossman_df_enc.loc[:,"Store"],
    rossman_df_enc.loc[:,"Sales"])

# %% Test train split
# rename for prophet
single_sales.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

# split last 10 days from shop==1
X_train = single_sales[single_sales.ds<='2014-07-20'][['ds', 'y']]
X_test = single_sales[single_sales.ds>'2014-07-20'][['ds', 'y']]

#%% Baseline model with mean

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train[["ds"]], X_train[["y"]])



y_predict = dummy_regr.predict(X_test["ds"])
mean_error = mean_squared_error(X_test["y"], y_predict)
mean_error


#%% Prophet



# fit
model = Prophet()
model.fit(X_train)

# check the predictions for the training data
pred_train = model.predict(X_train)

# use the trained model to make a forecast
pred_test = model.predict(X_test)

p = model.plot(pred_test)
