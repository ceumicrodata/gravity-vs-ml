# import packages
import pandas as pd
import os
import numpy as np
import datetime
import csv

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Path to the folder containing the chunked data
folder_path = "/gcs/gnn_chapter/trade_data"

# List of dataframes to store the data from each file
chunks = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a CSV file
    if filename.endswith(".csv"):
        # Load the file into a dataframe
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        # Append the dataframe to the list
        chunks.append(df)

data_out = pd.DataFrame()
best_params = []

for chunk in chunks:

    regr = RandomForestRegressor(n_jobs=-1, verbose=0, random_state=4000)

    tune_grid = {"max_depth": [10], "max_features": [30]}

    rf_random = GridSearchCV(
        regr,
        tune_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        verbose=3,
        n_jobs = -1
    )

    data = chunk.copy() 

    # Fill missing citynum 
    data.fillna(1, inplace=True)

    # Create lagged values of trade and shift target
    data['lag1_value'] = data.groupby(['iso_o', 'iso_d'])['Value'].shift(1)
    data['Value_target'] = data.groupby(['iso_o', 'iso_d'])['Value'].shift(-1)

    # keep the last year 
    X_predic = data[data['Period'] == max(data['Period'])].drop(['Value_target'], axis=1)

    # Drop because of shift
    data.dropna(inplace=True)
    
    # Run RF with 5 fold CV + grid search
    X = data.drop(['Value_target'], axis=1)
    y = data['Value_target']

    # use the specification from gravity
    X = X[['iso_o', 'iso_d','Period','contig', 'comlang_off','comcol', 'dist', 'Value', 'gdp_o', 'gdp_d', 'lag1_value']]
    X_predic = X_predic[['iso_o', 'iso_d','Period','contig', 'comlang_off','comcol', 'dist', 'Value', 'gdp_o', 'gdp_d', 'lag1_value']]
    
    # Exclude columns
    columns_to_exclude = ['iso_o', 'iso_d', 'Period']

    rf_random.fit(X.drop(columns=columns_to_exclude), y)
    y_predic = rf_random.predict(X_predic.drop(columns=columns_to_exclude))

    X_predic['prediction'] = y_predic
    data_out = pd.concat([data_out, X_predic])

    best_params.append(rf_random.best_params_)

# Get the current date and time
current_time = datetime.datetime.now()

# Format the timestamp as desired
timestamp = current_time.strftime("%Y-%m-%d_%H-%M")

# Use the timestamp when saving the DataFrame
filename = f"RF_trade_limited_lags_{timestamp}.csv"

# Save results
results = pd.DataFrame({
    'year': data_out['Period']+1,
    'iso_o': data_out['iso_o'],
    'iso_d': data_out['iso_d'],
    'prediction': data_out['prediction']
})

results.to_csv('/gcs/gnn_chapter/trade_results/' + filename, index=False)