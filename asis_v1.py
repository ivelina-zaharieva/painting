# %% Libraries
import pandas as pd
import numpy as np
from utils import *

!pwd

# %% Load TEMP
store = pd.read_csv("./painting/data/store.csv", dtype=str)
sales = pd.read_csv("./painting/data/train.csv", dtype=str)

data = pd.merge(sales, store, how='left', on='Store')

data.head(3)
data.dtypes


# %%
convert_vars(data)
