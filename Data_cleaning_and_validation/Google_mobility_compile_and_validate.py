# This code merges and validates the datasets for Google mobility prediction

import pandas as pd

###############
# Import datasets
###############

# State name mapping table
state_mapping_table = pd.read_csv('../Input_datasets/Mobility_flow_prediction_shared/us_states_w_fips_and_coordinates.csv')
# Drop US-DC (District of Columbia) as it is not a state
state_mapping_table = state_mapping_table[state_mapping_table["iso_3166_2_code"]!="US-DC"]
iso_2_fips_mapper = dict(zip(state_mapping_table['iso_3166_2_code'], state_mapping_table['FIPS']))
short_iso_2_fips_mapper = dict(zip(state_mapping_table['iso_3166_2_code'].apply(lambda x: x[3:]), state_mapping_table['FIPS']))
fips_iso_2_mapper = dict(zip(state_mapping_table['FIPS'], state_mapping_table['iso_3166_2_code']))

# Google Mobility dataset
google_mobility_dataset = pd.read_csv('../Input_datasets/Google_mobility_flow_prediction/Google_mobility_data.csv')
# Fill NA for missing targets
google_mobility_dataset[['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
               'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
               'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']] = \
               google_mobility_dataset[['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
               'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
               'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']].fillna(0)

# Import US state edgelist
edge_list = pd.read_json('../Input_datasets/Mobility_flow_prediction_shared/us_states_edge_list.json')
edge_list.columns= ['origin', 'destination']
# Drop FIPS=11 (US-DC, District of Columbia) as it is not a state
edge_list = edge_list[(edge_list['origin']!=11) & (edge_list['destination']!=11)]
# Add neighbouring column
edge_list['neighbouring'] = 1
# Create full edge list
full_edge_list = pd.DataFrame({(i, j) for i in edge_list['destination'].unique() for j in edge_list['destination'].unique()}, columns=['origin', 'destination'])
edge_list = pd.merge(full_edge_list, edge_list, on=['origin', 'destination'], how='left')
# Fill NA for neighbouring column
edge_list = edge_list.fillna(0)

# Additional edge characteristics
us_state_distances = pd.read_csv('../Input_datasets/Mobility_flow_prediction_shared/US_state_distances.csv', skiprows=2, index_col=0)
us_state_distances.columns = us_state_distances.index
us_state_distances = us_state_distances.drop(["DC", "PR"], axis=0)
us_state_distances = us_state_distances.drop(["DC", "PR"], axis=1)
us_state_distances.index = [short_iso_2_fips_mapper[x] for x in us_state_distances.index]
us_state_distances.columns = [short_iso_2_fips_mapper[x] for x in us_state_distances.columns]

# Node characteristics - US state population
us_state_pop = pd.read_csv('../Input_datasets/Mobility_flow_prediction_shared/US_state_pop_2019_census.csv', index_col=0)
us_state_pop.columns = ["state", "population_2019", "population_density_2019", "FIPS"]
us_state_pop.drop(columns=["state"], inplace=True)

# Node characteristics - OpenStreetMap features
overpass_features = pd.read_csv('../Input_datasets/Mobility_flow_prediction_shared/overpass_features.csv', index_col=0)
overpass_features.drop(columns=["overpass_id", "state"], inplace=True)
overpass_features.rename(columns={"state_short":"iso_3166_2_code"}, inplace=True)

# US state size
us_state_size = pd.read_csv('../Input_datasets/Mobility_flow_prediction_shared/US_state_sizes.csv', index_col=0)

###############
# Create node_list data
###############
node_list = pd.merge(state_mapping_table, us_state_pop, on = "FIPS", how='left')
node_list = pd.merge(node_list, overpass_features, on="iso_3166_2_code", how='left')
node_list = pd.merge(node_list, us_state_size, on="FIPS", how='left')

###############
# Drop Alaska and Hawaii from node list
###############
node_list = node_list[(node_list['iso_3166_2_code']!='US-AK') &
                      (node_list['iso_3166_2_code']!='US-HI')]

node_list.rename(columns={"iso_3166_2_code":"origin"}, inplace=True)

###############
# Create edge_list data
###############
edge_list["distances"] = edge_list.apply(lambda x: us_state_distances.loc[x["origin"], x["destination"]], axis=1)
edge_list['origin'] = edge_list['origin'].astype('int').apply(lambda x: fips_iso_2_mapper[x])
edge_list['destination'] = edge_list['destination'].astype('int').apply(lambda x: fips_iso_2_mapper[x])

###############
# Drop edges where origin = destination
###############

edge_list = edge_list[edge_list['origin'] != edge_list['destination']]

###############
# Create node_target_list data
###############
node_target_list = google_mobility_dataset.drop(columns=["country_region_code", "country_region", "sub_region_1"])

###############
# Drop Alaska and Hawaii from node target list
###############
node_target_list = node_target_list[(node_target_list['iso_3166_2_code']!='US-AK') &
                      (node_target_list['iso_3166_2_code']!='US-HI')]

###############
# Add timeline column which will be used as timestamp
###############
timeline_dict = dict(enumerate(sorted(node_target_list['date'].unique())))
timeline_mapping_dict = {y: x for x, y in timeline_dict.items()}

node_target_list['Timeline'] = node_target_list['date'].apply(lambda x: timeline_mapping_dict[x])

node_target_list.rename(columns={"iso_3166_2_code":"origin"}, inplace=True)

###############
# Melt node target list
###############
node_target_list_long = pd.melt(node_target_list, id_vars=["origin", "date", "Timeline"],
                           value_vars=['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
               'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
               'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline'], var_name='destination', value_name='Value')

###############
# Add flow value to other establishment types
###############
node_target_list_long = pd.merge(node_target_list_long, node_target_list, on=["origin", "date", "Timeline"])

###############
# Save to csv
###############
node_list.to_csv("../Output_datasets/Google_mobility_flow_prediction/node_list.csv", index=False)
edge_list.to_csv("../Output_datasets/Google_mobility_flow_prediction/edge_list.csv", index=False)
node_target_list_long.to_csv("../Output_datasets/Google_mobility_flow_prediction/node_target_list.csv", index=False)

# Node dataset
node_ids=['origin', 'destination']
node_timestamp='Timeline'
node_features=['population_2019', 'population_density_2019', 'Total_Area',
               'residential_land_use_area', 'commercial_land_use_area',
               'industrial_land_use_area', 'retail_land_use_area', 'natural_land_use_area',
	           'residential_roads_length', 'other_roads_length', 'main_roads_length',
		       'point_transport', 'building_transport', 'point_food', 'building_food',
	           'point_health', 'building_health', 'point_education', 'building_education',
               'point_retail', 'building_retail', 'date',
               'retail_and_recreation_percent_change_from_baseline',
               'grocery_and_pharmacy_percent_change_from_baseline',
               'parks_percent_change_from_baseline',
               'transit_stations_percent_change_from_baseline',
               'workplaces_percent_change_from_baseline',
               'residential_percent_change_from_baseline']

node_targets=['Value']

##################
#  Not used: Edge dataset
##################
flow_origin='origin'
flow_destination='destination'
#flows_value=''
flows_timestamp='date'
#flows_features=['neighbouring', 'distances']

# Chunk parameters
chunk_size = 350

# Chunked data path
chunk_path  = "../Output_datasets/Google_mobility_flow_prediction/Chunked_merged_data"

# Merge node_list and node_target_list
node_list = pd.merge(node_list, node_target_list_long, on=['origin'], how='inner')

# Filter only neccesary columns
nodes_columns = node_features + node_targets +node_ids + [node_timestamp]
nodes_columns = [i for i in nodes_columns if i!='']
node_list = node_list[nodes_columns]

nodes_and_edges = node_list.copy()

chunk_period_list = [nodes_and_edges[node_timestamp].unique()[i:i+chunk_size] for i in range(0, len(nodes_and_edges[node_timestamp].unique())-(chunk_size-1)) if i%50==0]
for chunk in chunk_period_list:
    nodes_and_edges_chunk = nodes_and_edges[nodes_and_edges[node_timestamp].isin(chunk)]
    chunk_name = str(min(chunk)) + "-" + str(max(chunk))
    nodes_and_edges_chunk.to_csv(f"{chunk_path}/{chunk_name}.csv", index=False)