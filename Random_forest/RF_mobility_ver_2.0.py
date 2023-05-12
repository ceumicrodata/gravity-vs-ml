# import packages
import pandas as pd
import os
import numpy as np
import sklearn 
from plotnine import ggplot, aes, geom_line
import math

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from skranger.ensemble import RangerForestRegressor
from sklearn.metrics import mean_squared_error

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

# Path to the folder containing the chunked data
folder_path = "../Output_datasets/Google_mobility_flow_prediction/Chunked_merged_data"

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

# Initialize h2o
h2o.init(nthreads=-1, max_mem_size = "32g")

# Loop through chunks
data_out = pd.DataFrame()

# Define model 
drf = H2ORandomForestEstimator(ntrees = 50, min_rows=10, nfolds = 5, stopping_metric= 'rmse', seed = 42)

# define the hyperparameter grid
tune_grid = {
    'max_depth': [15, 18, 20],
    'mtries': [21, 23]
    }

drf_grid = H2OGridSearch(model=drf,
                        hyper_params=tune_grid,
                        parallelism = 0
                        ) 
    
for chunk in chunks:

    data = chunk.copy()

    # define the list of features to be used 
    features = list(data.columns)
    features = features.remove ('Value')

    # Fill missing target (temporary solution)
    #data = data.bfill(axis ='rows')
    data['Value'] = data['Value'].fillna(0)

    # remove the separators from area and convert to numeric
    data['Total_Area'] = data['Total_Area'].str.replace(r',', r'', regex=True)
    data['Total_Area'] = pd.to_numeric(data['Total_Area'])


    # use categorical variables for origin and destination
    data['origin'] = data['origin'].astype('category')
    data['destination'] = data['destination'].astype('category')

    # Drop date
    data = data.drop('date', axis=1)

    # Create lag 
    data['lagged_Value'] = data.groupby(['origin', 'destination'])['Value'].shift(1)

    # Shift traget
    data['Value_target'] = data.groupby(['origin', 'destination'])['Value'].shift(-1)

    # keep the last year 
    X_predic = data[data['Timeline'] == max(data['Timeline'])].drop(['Value_target'], axis=1)

    # Drop because of shift
    data.dropna(inplace=True)

    # Convert to h2o dataframe
    data = h2o.H2OFrame(data)
    X_predic_h2o = h2o.H2OFrame(X_predic)

    # Dataset split
    data_split = data.split_frame(ratios=[0.8], seed = 42)
    data_train = data_split[0]
    data_test = data_split[1]

    # Train using 5 fold CV plus grid search and predict using best model 
    drf_grid.train(x = features, y = 'Value_target', training_frame = data_train, validation_frame = data_test)
    drf_sorted_grid = drf_grid.get_grid(sort_by = 'rmse', decreasing = False)
    best_model = drf_sorted_grid[0]
    y_predic_h2o = best_model.predict(X_predic_h2o)
    y_predic = y_predic_h2o.as_data_frame() 

    X_predic['prediction'] = y_predic['predict'].to_numpy()

    data_out = pd.concat([data_out, X_predic])

    # Save results
results = pd.DataFrame({
    'Timeline': data_out['Timeline']+1,
    'origin': data_out['origin'],
    'destination': data_out['destination'],
    'prediction': data_out['prediction']
})

results.to_csv('google_mobility_prediction_ver_2.0.csv')