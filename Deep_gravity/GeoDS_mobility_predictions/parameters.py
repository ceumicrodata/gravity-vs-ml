from ray import tune
from torch import nn
import numpy as np


##############
# Dataset params
##############
domain='GeoDS'
node_path = '../../Output_datasets/GeoDS_mobility_flow_prediction/node_list.csv'
node_target_path = ''
edge_path = '../../Output_datasets/GeoDS_mobility_flow_prediction/edge_list.csv'
edge_target_path='../../Output_datasets/GeoDS_mobility_flow_prediction/edge_target_list.csv'
chunk_path  = '../../Output_datasets/GeoDS_mobility_flow_prediction/Chunked_merged_data'
output_path = '../GeoDS_mobility_predictions/'

# Node dataset
node_id='iso_3166_2_code'
node_timestamp=''
node_features=['population_2019', 'population_density_2019',
               'residential_land_use_area', 'commercial_land_use_area',
               'industrial_land_use_area', 'retail_land_use_area', 'natural_land_use_area',
	           'residential_roads_length', 'other_roads_length', 'main_roads_length',
		       'point_transport', 'building_transport', 'point_food', 'building_food',
	           'point_health', 'building_health', 'point_education', 'building_education',
               'point_retail', 'building_retail']

node_targets = []

# Edge dataset
flow_origin='origin'
flow_destination='destination'
flows_value='pop_flows'
flows_timestamp='Timeline'
flows_features=['neighbouring', 'distances', 'visitor_flows']

# Chunk parameters
chunk_size = 50
window_size = 10
validation_period = 0.2 #10

# Add lag parameters
lag_periods = 1
time_dependent_edge_columns = ['pop_flows', 'visitor_flows']
time_dependent_node_columns = []

# Rename columns for final output
columns_to_rename = {'Timestamp_target':'year'}

# Log columns
columns_to_log = ['distances', 'visitor_flows', 'population_2019_o',
'population_density_2019_o', 'Total_Area_o', 'residential_land_use_area_o',
'commercial_land_use_area_o', 'industrial_land_use_area_o',
'retail_land_use_area_o', 'natural_land_use_area_o', 'residential_roads_length_o',
'other_roads_length_o', 'main_roads_length_o', 'point_transport_o',
'building_transport_o', 'point_food_o', 'building_food_o', 'point_health_o',
'building_health_o', 'point_education_o', 'building_education_o', 'point_retail_o',
'building_retail_o', 'population_2019_d', 'population_density_2019_d', 'Total_Area_d',
'residential_land_use_area_d', 'commercial_land_use_area_d', 'industrial_land_use_area_d',
'retail_land_use_area_d', 'natural_land_use_area_d', 'residential_roads_length_d',
'other_roads_length_d', 'main_roads_length_d', 'point_transport_d',
'building_transport_d', 'point_food_d', 'building_food_d', 'point_health_d',
'building_health_d', 'point_education_d', 'building_education_d',
'point_retail_d', 'building_retail_d']

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
        "dim_hidden": tune.sample_from(lambda _: 2**np.random.randint(3, 5)), #tune.sample_from(lambda _: 2**np.random.randint(2, 6)),
        "dropout_p": tune.choice([0.02, 0.05]), #tune.choice([0.25, 0.35, 0.45]),
        "num_layers": 5, #tune.choice([5, 10, 15]),
        "epochs": 1000, #tune.choice([500, 1000]),
        #"loss_fn": tune.choice([nn.L1Loss(), nn.MSELoss()])
    }

##############
# Model params
##############

max_epochs=1000
momentum=0.9
seed=1234
device='cpu'
loss_fn = nn.MSELoss()

# TBA add mode to only evaluate
#mode='train'
