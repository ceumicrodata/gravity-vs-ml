# import packages
import pandas as pd
import os
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Path to the folder containing the chunked data
folder_path = "/gcs/gnn_chapter/GeoDS_data"


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

    tune_grid = {"max_depth": [14], "max_features": [30]}

    rf_random = GridSearchCV(
        regr,
        tune_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        verbose=3,
    )
    
    data = chunk.copy()

    # remove the separators from area and convert to numeric
    data['Total_Area_o'] = data['Total_Area_o'].str.replace(r',', r'', regex=True)
    data['Total_Area_o'] = pd.to_numeric(data['Total_Area_o'])
    data['Total_Area_d'] = data['Total_Area_d'].str.replace(r',', r'', regex=True)
    data['Total_Area_d'] = pd.to_numeric(data['Total_Area_d'])

    # Drop date
    data = data.drop('start_date', axis=1)
    
    # Lag target
    lags = 3
    data = generate_lagged_columns(data, 'pop_flows', lags)

    # Shift traget
    data['pop_flows_target'] = data.groupby(['origin', 'destination'])['pop_flows'].shift(-1)

    # keep the last year 
    X_predic = data[data['Timeline'] == max(data['Timeline'])].drop(['pop_flows_target'], axis=1)

    # Drop because of shift
    data.dropna(inplace=True)

    # Run RF with 5 fold CV + grid search
    X = data.drop(['pop_flows_target'], axis=1)
    y = data['pop_flows_target']
    
    # use the specification from gravity 
    X = X[['origin', 'destination','Timeline','population_2019_o','population_2019_d', 'pop_flows', 'distances', 'neighbouring', 'all_pop_flows_to_d', 'o_pop_flows_to_all', 'lag1_pop_flows', 'lag2_pop_flows', 'lag3_pop_flows']]
    X_predic = X_predic[['origin', 'destination','Timeline','population_2019_o','population_2019_d', 'pop_flows', 'distances', 'neighbouring', 'all_pop_flows_to_d', 'o_pop_flows_to_all', 'lag1_pop_flows', 'lag2_pop_flows', 'lag3_pop_flows']]
    
    # Exclude columns
    columns_to_exclude = ['origin', 'destination', 'Timeline']
    
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
filename = f"RF_GeoDS_limited_lags_{timestamp}.csv"

# Save results
results = pd.DataFrame({
    'year': data_out['Timeline']+1,
    'origin': data_out['origin'],
    'destination': data_out['destination'],
    'prediction': data_out['prediction']
})

results.to_csv('/gcs/gnn_chapter/GeoDS_results/' + filename, index=False)
