from ray import tune
from torch import nn
import numpy as np

##############
# Dataset params
##############
domain='GeoDS'
node_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_nodelist.csv'
edge_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv'
edge_target_path = ''
#chunk_path  = '../../Output_datasets/Yearly_trade_data_prediction/Chunked_merged_data'
chunk_path = '/gcs/gnn_chapter/GeoDS_data'
output_path = '/gcs/gnn_chapter/GeoDS_results'


# Node dataset
node_id='iso_3166_2_code'
node_timestamp=''
node_features=['population_2019', 'population_density_2019', 'Total_area',
               'residential_land_use_area', 'commercial_land_use_area',
               'industrial_land_use_area', 'retail_land_use_area', 'natural_land_use_area',
               'residential_roads_length', 'other_roads_length', 'main_roads_length',
               'point_transport', 'building_transport', 'point_food', 'building_food',
               'point_health', 'building_health', 'point_education', 'building_education',
               'point_retail', 'building_retail']

# Edge dataset
flow_origin='origin'
flow_destination='destination'
flows_value='pop_flows'
flows_timestamp='Timeline'
flows_features=['neighbouring', 'distances', 'visitor_flows', 'visitor_flows_reverse',
                'pop_flows_reverse', 'all_visitor_flows_to_d', 'all_pop_flows_to_d',
                'all_visitor_flows_to_o', 'all_pop_flows_to_o', 'o_visitor_flows_to_all',
                'o_pop_flows_to_all', 'd_visitor_flows_to_all', 'd_pop_flows_to_all']

# Add lag parameters
lag_periods = 3
time_dependent_edge_columns = ['pop_flows'] #, 'visitor_flows', 'visitor_flows_reverse', 'pop_flows_reverse', 'all_visitor_flows_to_d', 'all_pop_flows_to_d', 'all_visitor_flows_to_o', 'all_pop_flows_to_o', 'o_visitor_flows_to_all', 'o_pop_flows_to_all', 'd_visitor_flows_to_all', 'd_pop_flows_to_all'
time_dependent_node_columns = []

# Rename columns for final output
columns_to_rename = {'Timestamp_target':'year'}

# Log columns
columns_to_scale = ['distances', 'visitor_flows', 'visitor_flows_reverse',
                'all_visitor_flows_to_d',
                'all_visitor_flows_to_o', 'o_visitor_flows_to_all',
                'd_visitor_flows_to_all',
                'population_2019', 'population_density_2019',
                'Total_Area', 'residential_land_use_area',
                'commercial_land_use_area', 'industrial_land_use_area',
                'retail_land_use_area', 'natural_land_use_area', 'residential_roads_length',
                'other_roads_length', 'main_roads_length', 'point_transport',
                'building_transport', 'point_food', 'building_food', 'point_health',
                'building_health', 'point_education', 'building_education', 'point_retail',
                'building_retail'] #pop_flows, 'pop_flows_reverse', 'all_pop_flows_to_d', 'all_pop_flows_to_o', 'o_pop_flows_to_all', 'd_pop_flows_to_all',

# Chunk parameters
validation_period = 0.3

##############
# Ray Tune params
##############
config = {
        "lr": 0.0005, #0.000075, #tune.choice([0.000025, 0.00005, 0.000075, 0.0001]), #tune.choice([0.00025, 0.0003]), #tune.choice([0.005, 0.005]), #tune.choice([0.005, 0.001]), #tune.choice([0.00025, 0.0003]), #tune.loguniform(1e-4, 1e-1),
        "batch_size": 32, #tune.choice([32, 64, 128]), #tune.choice([32, 64, 128]), #tune.choice([32, 64, 128]),
        "dim_hidden": 32, #tune.grid_search([32, 64]), #tune.sample_from(lambda _: 2**np.random.randint(3, 5)), #16, #tune.choice([8, 16, 32]), #tune.choice([16, 32]), #tune.sample_from(lambda _: 2**np.random.randint(3, 5)),
        "dropout_p": 0.02, #tune.choice([0.02, 0.05]), #tune.choice([0.02, 0.05]), 
        "num_layers": 3, #tune.grid_search([3, 4]), #tune.choice([4, 5]), #tune.choice([3, 4, 5]), #tune.choice([5, 10, 15]),
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

