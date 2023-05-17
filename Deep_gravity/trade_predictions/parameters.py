from ray import tune
from torch import nn
import numpy as np

##############
# Dataset params
##############
domain='trade'
node_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_nodelist.csv'
node_target_path = ''
edge_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv'
edge_target_path = ''
chunk_path  = '../../Output_datasets/Yearly_trade_data_prediction/Chunked_merged_data'
output_path = '../trade_predictions/'

# Node dataset
node_id='iso_numeric'
node_timestamp='year'
node_features=['gdp', 'total_population',
               'urban_population(%_of_total)',
               'area', 'dis_int', 'landlocked', 'citynum']

node_targets = []

# Edge dataset
flow_origin='iso_o'
flow_destination='iso_d'
flows_value='Value'
flows_timestamp='Period'
flows_features=['contig', 'comlang_off', 'comlang_ethno', 'colony',
                'comcol', 'curcol', 'col45', 'smctry', 'dist', 'distcap', 'distw', 'distwces']

# Chunk parameters
#chunk_size = 5
#window_size = 1
validation_period = 0.2

# Add lag parameters
lag_periods = 1
time_dependent_edge_columns = ['Value']
time_dependent_node_columns = ['gdp', 'total_population', 'urban_population(%_of_total)']

# Rename columns for final output
columns_to_rename = {'Timestamp_target':'year'}

# Log columns
columns_to_log = ['dist', 'distcap', 'distw', 'distwces', 'gdp_o', 'total_population_o',
'urban_population(%_of_total)_o', 'area_o', 'dis_int_o',
'gdp_d', 'total_population_d', 'urban_population(%_of_total)_d', 'area_d', 'dis_int_d',
'Value_1', 'gdp_o_1', 'gdp_d_1', 'total_population_o_1',  'total_population_d_1',
'urban_population(%_of_total)_o_1', 'urban_population(%_of_total)_d_1']

##############
# Global settings
##############
# Add different models and prediction types
# TBA: model_type = 'DeepGravity'
#TBA: prediction_type = 'node' or 'edge'


##############
# Ray Tune params
##############

config = {
        "lr": tune.choice([0.00025, 0.0003]), #tune.loguniform(1e-4, 1e-1),
        "batch_size": 32, #tune.choice([2, 4, 8, 16]),
        "dim_hidden": tune.sample_from(lambda _: 2**np.random.randint(3, 5)),
        "dropout_p": tune.choice([0.02, 0.05]),
        "num_layers": 5, #tune.choice([5, 10, 15]),
        "epochs": 1000, #tune.choice([500, 1000]),
        #"loss_fn": tune.choice([nn.L1Loss(), nn.MSELoss()])
    }

##############
# Model params
##############

#epochs=500
momentum = 0.9
seed = 1234
device = 'cpu'
loss_fn = nn.MSELoss()

#ASHAScheduler parameters
max_epochs = 1000
grace_period = 50
reduction_factor = 2

#Tune parameters
resources_per_trial = 4
num_samples = 8
weight_decay = 0
early_stopper_patience=10
early_stopper_min_delta=3e+17 #2e+8 for GeoDS
early_stopper_grace_period = 100

#Data_loader parameters
data_loader_batch_size = 32

# TBA add mode to only evaluate
#mode='train'
