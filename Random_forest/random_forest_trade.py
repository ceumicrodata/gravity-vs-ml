# import packages
import pandas as pd
import os
import numpy as np
import seaborn as sns
import missingno as msno
import sklearn 
from plotnine import ggplot, aes, geom_line

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from skranger.ensemble import RangerForestRegressor

# Load data
trade_nodelist = pd.read_csv('../Output_datasets/Yearly_trade_data_prediction/trade_nodelist.csv')
trade_edgelist = pd.read_csv('../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv')

# Drop lines where the origin is the same as destination
trade_edgelist = trade_edgelist.drop(trade_edgelist[trade_edgelist["iso_o"] == trade_edgelist["iso_d"]].index)

# Create lagged values of trade: Predict next year using the previous year instead of current
trade_edgelist['lag_value'] = trade_edgelist.groupby(['iso_o', 'iso_d'])['Value'].shift(1)
trade_edgelist['Value'] = trade_edgelist.groupby(['iso_o', 'iso_d'])['Value'].shift(-1)

# Drop variables missing too many observations
trade_nodelist.drop(['langoff_2', 'langoff_3', 'pays', 'lang20_1', 'lang20_2', 'lang20_3', 'lang20_4', 'lang9_1', 'lang9_2',
       'lang9_3', 'lang9_4', 'colonizer1', 'colonizer2', 'colonizer3',
       'colonizer4', 'short_colonizer1', 'short_colonizer2',
       'short_colonizer3'], axis=1, inplace=True)

# Merge once for origin and once for destination 
trade=pd.merge(trade_edgelist,trade_nodelist,how='left', left_on=['iso_o','Period'], right_on=['iso_numeric','year'],suffixes=('', '_o'))
trade=pd.merge(trade,trade_nodelist,how='left', left_on=['iso_d','Period'], right_on=['iso_numeric','year'],suffixes=('', '_d'))
trade

# Drop observations with missing data still
trade.dropna(inplace=True)

# Select features
data = trade[["Period", "Value", "iso_o", "iso_d", "contig", "comlang_off", "comlang_ethno", "colony", "comcol", "curcol", "col45", "smctry", "dist", "distcap",'distcap', 'distw', 'distwces',
       'year', 'gdp', 'total_population', 'urban_population(%_of_total)',
       'area', 'dis_int', 'landlocked',
       'citynum', 'gdp_d', 'total_population_d',
       'urban_population(%_of_total)_d', 'cnum_d',
       'area_d', 'dis_int_d', 'landlocked_d', 'citynum_d', 'lag_value']]

# Train using 5 years and predict one year plus grid search

regr = RangerForestRegressor(importance="impurity",max_depth=5, seed=42, n_jobs = -1)

tune_grid = {"mtry": [8, 10, 12], "min_node_size": [5, 10, 15]}

rf_random = GridSearchCV(
    regr,
    tune_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    verbose=3,
)

trade_out = pd.DataFrame()
for year in range(1996, 2014):
    print(year)
    data_train = data[(data["Period"] >= year) & (data["Period"] <= year + 4)]
    data_test = data[(data["Period"] == year + 5)]
    X_train= data_train.drop(['Value'], axis=1)
    X_test= data_test.drop(['Value'], axis=1)
    y_train = data_train['Value']
    y_test = data_test['Value']

    rf_random.fit(X_train, y_train)
    y_predic = rf_random.predict(X_test)

    X_test['target'] = y_test
    X_test ['prediction'] = y_predic
   
    trade_out = pd.concat([trade_out, X_test])

    # Save results
results = pd.DataFrame({
    'year': trade_out['Period'],
    'iso_o': trade_out['iso_o'],
    'iso_d': trade_out['iso_d'],
    'target': trade_out['target'],
    'prediction': trade_out['prediction']
})

results.to_csv('prediction.csv')
