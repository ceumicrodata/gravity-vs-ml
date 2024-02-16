from ray import tune
from torch import nn
import numpy as np


##############
# Dataset params
##############
domain='trade'
node_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_nodelist.csv'
edge_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv'
edge_target_path = ''
#chunk_path  = '../../Output_datasets/Yearly_trade_data_prediction/Chunked_merged_data'
chunk_path = '/gcs/gnn_chapter/trade_data'
output_path = '/gcs/gnn_chapter/trade_results'

# Node dataset
node_id='iso_numeric'
node_timestamp='year'
node_features=['gdp', 'total_population',
               'urban_population(%_of_total)',
               'area', 'dis_int', 'landlocked', 'citynum']

# Edge dataset
flow_origin='iso_o'
flow_destination='iso_d'
flows_value='Value'
flows_timestamp='Period'
flows_features=['contig', 'comlang_off', 'comlang_ethno', 'colony',
                'comcol', 'curcol', 'col45', 'smctry', 'dist', 'distcap', 'distw', 'distwces',
                'Value_reverse', 'all_to_d', 'all_to_o', 'o_to_all', 'd_to_all']

# Add lag parameters
lag_periods = 1
time_dependent_edge_columns = ['Value'] #, 'Value_reverse', 'all_to_d', 'all_to_o', 'o_to_all', 'd_to_all'
time_dependent_node_columns = [] #'gdp', 'total_population', 'urban_population(%_of_total)'

# Rename columns for final output
columns_to_rename = {'Timestamp_target':'year'}

# Log columns
columns_to_scale = ['gdp', 'total_population', 'urban_population(%_of_total)', 'area', 'dis_int', 'dist', 'distcap', 'distw', 'distwces', 'Value_reverse', 'all_to_d', 'all_to_o', 'o_to_all', 'd_to_all'] #Period, Value,

# Chunk parameters
validation_period = 0.3

##############
# Ray Tune params
##############
config = {
        "lr": 0.00005, #0.000075, #tune.choice([0.000025, 0.00005, 0.000075, 0.0001]), #tune.choice([0.00025, 0.0003]), #tune.choice([0.005, 0.005]), #tune.choice([0.005, 0.001]), #tune.choice([0.00025, 0.0003]), #tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.grid_search([32]), #32, #tune.choice([32, 64, 128]), #tune.choice([32, 64, 128]), #tune.choice([32, 64, 128]),
        "dim_hidden": tune.grid_search([32]), #tune.sample_from(lambda _: 2**np.random.randint(3, 5)), #16, #tune.choice([8, 16, 32]), #tune.choice([16, 32]), #tune.sample_from(lambda _: 2**np.random.randint(3, 5)),
        "dropout_p": 0.02, #tune.choice([0.02, 0.05]), #tune.choice([0.02, 0.05]), 
        "num_layers": tune.grid_search([3]), #tune.choice([4, 5]), #tune.choice([3, 4, 5]), #tune.choice([5, 10, 15]),
        "epochs": 2000, #tune.choice([500, 1000]),
    }

##############
# Model params
##############
#epochs=500
momentum = 0.9
seed = 4000
device = 'cpu'
# Loss function
loss_fn = nn.MSELoss()
# Number of CPUs per trial
resources_per_trial = 10
# Number of hyperparameter combinations to try
num_samples = 1
# Weight decay used in RMSprop optimizer
weight_decay = 0
# Parameters of custom early stopper function
early_stopper_patience=5
early_stopper_min_delta=3e+17 #2e+8 for GeoDS
early_stopper_grace_period = 100

##############
# ASHAScheduler parameters
##############
max_epochs = 2000
grace_period = 50
reduction_factor = 2

# TBA add mode to only evaluate
# mode='train'

