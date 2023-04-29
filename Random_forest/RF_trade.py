# import packages
import pandas as pd
import os
import numpy as np
import seaborn as sns
import missingno as msno
import sklearn 
from plotnine import ggplot, aes, geom_line
import math

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from skranger.ensemble import RangerForestRegressor
from sklearn.metrics import mean_squared_error

# Path to the folder containing the chunked data
folder_path = "../Output_datasets/Yearly_trade_data_prediction/Chunked_merged_data"

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
for chunk in chunks:

    regr = RangerForestRegressor(importance="impurity", seed=42, n_jobs = -1)

    tune_grid = {"mtry": [15, 20, 25], "max_depth": [4, 7, 10]}

    rf_random = GridSearchCV(
        regr,
        tune_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        verbose=3,
    )

    data = chunk.copy()

    # Fill missing citynum 
    data.fillna(1, inplace=True)

    # Create lagged values of trade and shift target
    data['lag_value'] = data.groupby(['iso_o', 'iso_d'])['Value'].shift(1)
    data['Value_target'] = data.groupby(['iso_o', 'iso_d'])['Value'].shift(-1)
    # keep the last year 
    X_predic = data[data['Period'] == max(data['Period'])].drop(['Value_target'], axis=1)

    # Drop because of shift
    data.dropna(inplace=True)
    
    # Run RF with 5 fold CV + grid search
    X = data.drop(['Value_target'], axis=1)
    y = data['Value_target']

    rf_random.fit(X, y)
    y_predic = rf_random.predict(X_predic)

    X_predic ['prediction'] = y_predic


    data_out = pd.concat([data_out, X_predic])

    # Save results
results = pd.DataFrame({
    'year': data_out['Period']+1,
    'iso_o': data_out['iso_o'],
    'iso_d': data_out['iso_d'],
    'prediction': data_out['prediction']
})

results.to_csv('trade_prediction.csv')