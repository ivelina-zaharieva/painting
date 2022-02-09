# %% -- Convert VARS

def convert_vars(df):
    """
     Input: Rossman df
     Output: same df with re-formated variable types
     """
    df = data
    df.dtypes
    # set vars
    to_num = ["Sales", "CompetitionDistance"]
    to_cat = ["Store", "DayOfWeek", "Open",
              "Promo", "SchoolHoliday", "StateHoliday"]
    df["Date"] = pd.to_datetime(df["Date"])
    #date_expand(df)

    # Loop
    for i in to_cat:
         df.loc[:, i] = pd.Categorical(df[i])
     for i in to_num:
         df.loc[:, i] = pd.to_numeric(df[i], downcast='float', errors='coerce')

     return df


# %% -- Sanity check function
def cleanup(df):
    """
    Function description
    """
    df = df[~((df.Sales < 1) & (df.Open == 1))]  # drop bad values
    df = df[~(df["Open"] == 0)]  # drop days with shops closed
    df = df.drop(["Open"], axis=1)  # drop col

    return df


# %% -- Encoding

def date_expand(df):
    """
    Input: a df with column "Date"
    Out: expands multiple datetime features to columns
    """
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['WeekOfYear'] = df.Date.dt.weekofyear
    return df


def var_encoding(df):
    """
    input: dataframe from Rossman
    output: df with new datetime features
    """
    date_expand(df)  # expand datetime info

    # Target encoding
    t_enc = TargetEncoder()
    df["Assortment_encoded"] = t_enc.fit_transform(
        train_dataset.loc[:, "Assortment"],
        train_dataset.loc[:, "Sales"])

    # Mean encoding
    avg_1 = pd.DataFrame(train_dataset.groupby(
     ["Store", "DayOfWeek"]).mean()["Customers_x"]).reset_index()
    avg_1.rename(columns={'Customers_x': 'avg_daily_customers'}, inplace=True)

    # Merge
    pd.merge(df, avg_1, lef_on=['Store', 'DayOfWeek'])

    return df
