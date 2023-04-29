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
chunk_size = 5
window_size = 1
validation_period = 1

# Add lag parameters
lag_periods = 1
time_dependent_edge_columns = ['Value']
time_dependent_node_columns = ['gdp', 'total_population', 'urban_population(%_of_total)']

# Rename columns for final output
columns_to_rename = {'Timestamp_target':'year'}

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
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "dim_hidden": tune.sample_from(lambda _: 2**np.random.randint(1, 4)),
        "dropout_p": tune.choice([0.05, 0.15, 0.25]),
        "num_layers": tune.choice([2, 5, 10]),
    }

##############
# Model params
##############

epochs=500
momentum=0.9
seed=1234
device='cpu'
loss_fn = nn.L1Loss()

# TBA add mode to only evaluate
#mode='train'
