from ray import tune
from torch import nn
import numpy as np

##############
# Dataset params
##############
domain='Google'
node_path = '../../Output_datasets/Google_mobility_flow_prediction/node_list.csv'
node_target_path = '../../Output_datasets/Google_mobility_flow_prediction/node_target_list.csv'
edge_path = '../../Output_datasets/Google_mobility_flow_prediction/edge_list.csv'
edge_target_path=''
chunk_path  = '../../Output_datasets/Google_mobility_flow_prediction/Chunked_merged_data'
output_path = '../google_mobility_predictions/'

# Node dataset
node_id='iso_3166_2_code'
node_timestamp='Timeline'
node_features=['population_2019', 'population_density_2019',
               'residential_land_use_area', 'commercial_land_use_area',
               'industrial_land_use_area', 'retail_land_use_area', 'natural_land_use_area',
	           'residential_roads_length', 'other_roads_length', 'main_roads_length',
		       'point_transport', 'building_transport', 'point_food', 'building_food',
	           'point_health', 'building_health', 'point_education', 'building_education',
               'point_retail', 'building_retail']

node_targets = ['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
               'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
               'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']

# Edge dataset
flow_origin='origin'
flow_destination='destination'
flows_value=''
flows_timestamp=''
flows_features=['neighbouring', 'distances']

# Chunk parameters
chunk_size = 350
window_size = 50
validation_period = 10

# Add lag parameters
lag_periods = 1
time_dependent_edge_columns = ['Value']
time_dependent_node_columns = []

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
        "dim_hidden": tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "dropout_p": tune.choice([0.25, 0.35, 0.45]),
        "num_layers": tune.choice([5, 10, 15]),
    }

##############
# Model params
##############

epochs=10
momentum=0.9
seed=1234
device='cpu'
loss_fn = nn.MSELoss()

# TBA add mode to only evaluate
#mode='train'
