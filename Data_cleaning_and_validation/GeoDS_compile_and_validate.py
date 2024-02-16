# This code merges and validates the datasets used for GeoDS mobility prediction

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

# GeoDS Mobility dataset
geods_mobility_dataset = pd.read_csv('../Input_datasets/GeoDS_mobility_flow_prediction/state2state_merged.csv')
# Drop Puerto Rico and US-DC (District of Columbia)
geods_mobility_dataset = geods_mobility_dataset[(geods_mobility_dataset['geoid_o']!=11) & (geods_mobility_dataset['geoid_d']!=11) &
                                                (geods_mobility_dataset['geoid_o']!=72) & (geods_mobility_dataset['geoid_d']!=72)]

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
# Fill NA for eighbouring column
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

state_mapping_table = state_mapping_table[state_mapping_table["iso_3166_2_code"]!="US-DC"]
node_list = pd.merge(state_mapping_table, us_state_pop, on = "FIPS", how='left')
node_list = pd.merge(node_list, overpass_features, on="iso_3166_2_code", how='left')
node_list = pd.merge(node_list, us_state_size, on="FIPS", how='left')

###############
# Drop Alaska and Hawaii from node list
###############
node_list = node_list[(node_list['iso_3166_2_code']!='US-AK') &
                      (node_list['iso_3166_2_code']!='US-HI')]

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
# Create edge_target_list data
###############
edge_target_list = geods_mobility_dataset.copy()
edge_target_list['geoid_o'] = edge_target_list['geoid_o'].astype('int').apply(lambda x: fips_iso_2_mapper[x])
edge_target_list['geoid_d'] = edge_target_list['geoid_d'].astype('int').apply(lambda x: fips_iso_2_mapper[x])
edge_target_list.rename(columns={"geoid_o":"origin", "geoid_d": "destination"}, inplace=True)

###############
# Drop edges where origin = destination
###############

edge_target_list = edge_target_list[edge_target_list['origin'] != edge_target_list['destination']]

###############
# Drop Alaska and Hawaii from edge target list
###############
edge_target_list = edge_target_list[(edge_target_list['origin']!='US-AK') &
                      (edge_target_list['origin']!='US-HI') &
                      (edge_target_list['destination']!='US-AK') &
                      (edge_target_list['destination']!='US-HI')]


###############
# Add timeline column which will be used as timestamp
###############
timeline_dict = dict(enumerate(sorted(edge_target_list['start_date'].unique())))
timeline_mapping_dict = {y: x for x, y in timeline_dict.items()}

edge_target_list['Timeline'] = edge_target_list['start_date'].apply(lambda x: timeline_mapping_dict[x])

###############
# Validate mobility data
###############

# Obtain list of countries from trade dataset and validate

origin_set = set(edge_target_list.origin.unique())
destination_set = set(edge_target_list.destination.unique())

if (origin_set - destination_set != set()) & (destination_set - origin_set != set()):
    print('Number of partners and reporters do no match!')

periods = set(edge_target_list.start_date.unique())
all_pairs = set([(i,j,k) for i in origin_set for j in destination_set for k in periods])
real_pairs = set(list(edge_target_list[['origin', 'destination', 'start_date']].itertuples(index=False, name=None)))

if (all_pairs - real_pairs != set()) & (real_pairs - all_pairs != set()):
    print('Number of expected and real observations do no match!')

###############
# Add reverse flow value
###############

edge_target_list_reverse = edge_target_list[["start_date", "Timeline",  "visitor_flows", "pop_flows", "origin", "destination"]].copy()
edge_target_list_reverse.rename(columns = {"visitor_flows": "visitor_flows_reverse", "pop_flows": "pop_flows_reverse",
                                           "origin":"destination", "destination":"origin"}, inplace=True)

edge_target_list = pd.merge(edge_target_list, edge_target_list_reverse, on=["start_date", "Timeline",  "origin", "destination"], how="left")

###############
# All to node
###############
all_to_node = edge_target_list.groupby(["start_date", "Timeline",  "destination"])[["visitor_flows", "pop_flows"]].sum().reset_index()

all_to_node.rename(columns={"visitor_flows": "all_visitor_flows_to_d",
                            "pop_flows": "all_pop_flows_to_d"}, inplace=True)

edge_target_list = pd.merge(edge_target_list, all_to_node, on=["start_date", "Timeline",  "destination"], how="left")

all_to_node.rename(columns={"all_visitor_flows_to_d": "all_visitor_flows_to_o",
                            "all_pop_flows_to_d": "all_pop_flows_to_o", "destination":"origin"}, inplace=True)

edge_target_list = pd.merge(edge_target_list, all_to_node, on=["start_date", "Timeline", "origin"], how="left")

###############
# Node to all
###############
node_to_all = edge_target_list.groupby(["start_date", "Timeline", "origin"])[["visitor_flows", "pop_flows"]].sum().reset_index()

node_to_all.rename(columns={"visitor_flows": "o_visitor_flows_to_all",
                            "pop_flows": "o_pop_flows_to_all",}, inplace=True)

edge_target_list = pd.merge(edge_target_list, node_to_all, on=["start_date", "Timeline", "origin"], how="left")

node_to_all.rename(columns={"o_visitor_flows_to_all": "d_visitor_flows_to_all",
                            "o_pop_flows_to_all": "d_pop_flows_to_all", "origin":"destination"}, inplace=True)

edge_target_list = pd.merge(edge_target_list, node_to_all, on=["start_date", "Timeline", "destination"], how="left")

###############
# Save to csv
###############
node_list.to_csv("../Output_datasets/GeoDS_mobility_flow_prediction/node_list.csv", index=False)
edge_list.to_csv("../Output_datasets/GeoDS_mobility_flow_prediction/edge_list.csv", index=False)
edge_target_list.to_csv("../Output_datasets/GeoDS_mobility_flow_prediction/edge_target_list.csv", index=False)

# Node dataset
node_id='iso_3166_2_code'
node_timestamp=''
node_features=['population_2019', 'population_density_2019','Total_Area',
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
flows_features=['neighbouring', 'distances', 'visitor_flows', 'start_date',
                'visitor_flows_reverse', 'pop_flows_reverse',
                'all_visitor_flows_to_d', 'all_pop_flows_to_d',
                'all_visitor_flows_to_o', 'all_pop_flows_to_o',
                'o_visitor_flows_to_all', 'o_pop_flows_to_all',
                'd_visitor_flows_to_all', 'd_pop_flows_to_all']

# Chunk parameters
chunk_size = 50

# Chunked data path
chunk_path  = "../Output_datasets/GeoDS_mobility_flow_prediction/Chunked_merged_data"

# Merge edge_list and edge_target_list
edge_list = pd.merge(edge_list, edge_target_list, on=['origin', 'destination'], how='inner')

# Filter only neccesary columns
nodes_columns = node_features + [node_id] + [node_timestamp]
nodes_columns = [i for i in nodes_columns if i!='']
node_list = node_list[nodes_columns]

edges_columns = flows_features + [flow_origin] + [flow_destination] + \
    [flows_timestamp] + [flows_value]
edges_columns = [i for i in edges_columns if i!='']
edge_list = edge_list[edges_columns]

# Merge nodes and edges
node_list.rename(columns={node_id: flow_origin, node_timestamp: flows_timestamp}, inplace=True)
node_list[flow_destination] = node_list[flow_origin]

nodes_and_edges = pd.merge(pd.merge(edge_list, node_list.drop(flow_destination, axis=1), how='left', on=[flow_origin]),
                        node_list.drop(flow_origin, axis=1), how='left', on=[flow_destination], 
                        suffixes=('_o', '_d'))

chunk_period_list = [nodes_and_edges[flows_timestamp].unique()[i:i+chunk_size] for i in range(0, len(nodes_and_edges[flows_timestamp].unique())-(chunk_size-1)) if i%10==0]
for chunk in chunk_period_list:
    nodes_and_edges_chunk = nodes_and_edges[nodes_and_edges[flows_timestamp].isin(chunk)]
    chunk_name = str(min(chunk)) + "-" + str(max(chunk))
    nodes_and_edges_chunk.to_csv(f"{chunk_path}/{chunk_name}.csv", index=False)
