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
folder_path = "/gcs/gnn_chapter/Google_data"

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

# Create lags
def generate_lagged_columns(df, variable, lags):
    for i in range(1, lags + 1):
        lagged_var_name = f'lag{i}_{variable}'
        df[lagged_var_name] = df.groupby(['origin', 'destination'])[variable].shift(i)
    return df

for chunk in chunks:

    regr = RandomForestRegressor(random_state=4000, n_jobs=-1, verbose=0)

    tune_grid = {"max_depth": [28], "max_features": [18]} #{"max_depth": [10, 12, 14, 16, 20, 24, 28, 32], "max_features": [1, 2, 4, 6, 10, 14, 18, 22]}

    rf_random = GridSearchCV(
        regr,
        tune_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        verbose=3,
    )

    data = chunk.copy()

    # Fill missing target (temporary solution)
    #data = data.bfill(axis ='rows')
    data['Value'] = data['Value'].fillna(0)

    # remove the separators from area and convert to numeric
    data['Total_Area'] = data['Total_Area'].str.replace(r',', r'', regex=True)
    data['Total_Area'] = pd.to_numeric(data['Total_Area'])

    # Drop date
    data = data.drop('date', axis=1)

    # Lag target
    lags = 10
    data = generate_lagged_columns(data, 'Value', lags)

    # Shift traget
    data['Value_target'] = data.groupby(['origin', 'destination'])['Value'].shift(-1)

    # define the list of features to be used
    features = list(data.columns)
    features.remove ('Value_target')

    # keep the last year
    X_predic = data[data['Timeline'] == max(data['Timeline'])].drop(['Value_target'], axis=1)

    # Drop because of shift
    data.dropna(inplace=True)

    # Run RF with 5 fold CV + grid search
    X = data.drop(['Value_target'], axis=1)
    y = data['Value_target']

    # use the specification from gravity 
    X = X[['origin', 'destination', 'Timeline', 'population_2019', 'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline', 'Value']]
    X_predic = X_predic[['origin', 'destination', 'Timeline', 'population_2019', 'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline', 'Value']]
    
    # Exclude columns
    columns_to_exclude = ['origin', 'destination', 'Timeline']
    
    rf_random.fit(X.drop(columns=columns_to_exclude), y)
    y_predic = rf_random.predict(X_predic.drop(columns=columns_to_exclude))
    
    X_predic ['prediction'] = y_predic
    data_out = pd.concat([data_out, X_predic])

    best_params.append(rf_random.best_params_)


# Get the current date and time
current_time = datetime.datetime.now()

# Format the timestamp as desired
timestamp = current_time.strftime("%Y-%m-%d_%H-%M")

# Use the timestamp when saving the DataFrame
filename = f"RF_Google_limited_{timestamp}.csv"

# Save results
results = pd.DataFrame({
    'year': data_out['Timeline']+1,
    'origin': data_out['origin'],
    'destination': data_out['destination'],
    'prediction': data_out['prediction']
})

results.to_csv('/gcs/gnn_chapter/Google_results/' + filename, index=False)


