from ray import tune
from torch import nn
import numpy as np

##############
# Dataset params
##############
domain='Google'
node_path = '../../Output_datasets/Google_mobility_flow_prediction/node_list.csv'
edge_path = ''
edge_target_path='../../Output_datasets/Google_mobility_flow_prediction/node_target_list.csv'
#chunk_path  = '../../Output_datasets/Google_mobility_flow_prediction/Chunked_merged_data'
#output_path = '../google_mobility_predictions/'
chunk_path = '/gcs/gnn_chapter/Google_data'
output_path = '/gcs/gnn_chapter/Google_results'

# Node dataset
node_id='origin'
node_timestamp=''
node_features=['population_2019', 'population_density_2019', 'Total_Area',
               'residential_land_use_area', 'commercial_land_use_area',
               'industrial_land_use_area', 'retail_land_use_area', 'natural_land_use_area',
	           'residential_roads_length', 'other_roads_length', 'main_roads_length',
		       'point_transport', 'building_transport', 'point_food', 'building_food',
	           'point_health', 'building_health', 'point_education', 'building_education',
               'point_retail', 'building_retail']

# Edge dataset
flow_origin='origin'
flow_destination='destination'
flows_value='Value'
flows_timestamp='Timeline'
flows_features=['retail_and_recreation_percent_change_from_baseline',
                'grocery_and_pharmacy_percent_change_from_baseline',
                'parks_percent_change_from_baseline',
                'transit_stations_percent_change_from_baseline',
                'workplaces_percent_change_from_baseline',
                'residential_percent_change_from_baseline']

# Add lag parameters
lag_periods = 10
time_dependent_edge_columns = ['Value'] #, 'retail_and_recreation_percent_change_from_baseline',
    #'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline',
    #'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
    #'residential_percent_change_from_baseline'
time_dependent_node_columns = []

# Rename columns for final output
columns_to_rename = {'Timestamp_target':'year'}

# Log columns
columns_to_scale = ['population_2019', 'population_density_2019', 'Total_Area',
               'residential_land_use_area', 'commercial_land_use_area',
               'industrial_land_use_area', 'retail_land_use_area', 'natural_land_use_area',
	           'residential_roads_length', 'other_roads_length', 'main_roads_length',
		       'point_transport', 'building_transport', 'point_food', 'building_food',
	           'point_health', 'building_health', 'point_education', 'building_education',
               'point_retail', 'building_retail'] #'Timeline', 

# Chunk parameters
validation_period = 0.3

##############
# Ray Tune params
##############

config = {
        "lr": 0.0005,#tune.choice([0.00015, 0.00025, 0.0003]), #tune.loguniform(1e-4, 1e-1),
        "batch_size": 32, #tune.choice([2, 4, 8, 16]),
        "dim_hidden": 32, #tune.sample_from(lambda _: 2**np.random.randint(3, 5)), #tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "dropout_p": 0.02, #tune.choice([0.02, 0.05]), #tune.choice([0.25, 0.35, 0.45]),
        "num_layers": 3,#tune.choice([2, 3, 4, 5]), #tune.choice([5, 10, 15]),
        "epochs": 2000, #tune.choice([500, 1000]),
        #"loss_fn": tune.choice([nn.L1Loss(), nn.MSELoss()])
    }

##############
# Model params
##############
#epochs=10
momentum=0.9
seed=4000
device='cpu'
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
early_stopper_min_delta=20
early_stopper_grace_period = 100

##############
# ASHAScheduler parameters
##############
#
max_epochs = 2000
grace_period = 50
reduction_factor = 2

# TBA add mode to only evaluate
#mode='train'

