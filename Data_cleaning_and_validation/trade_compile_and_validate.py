# This code merges and validates the datasets used for trade modelling

import pandas as pd

###############
# Import datasets
###############

# Country name mapping table
country_mapping_table = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/country_codes.csv')
iso_alpha3_numeric_mapper = dict(zip(country_mapping_table['ISO3166-1-Alpha-3'], country_mapping_table['ISO3166-1-numeric']))
iso_alpha2_numeric_mapper = dict(zip(country_mapping_table['ISO3166-1-Alpha-2'], country_mapping_table['ISO3166-1-numeric']))

# Trade dataset
trade_dataset = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/trade_data_new_annual_import_zero_padded.csv')
country_names_in_trade_dataset = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/country_names_with_annual_trade_data.csv')
trade_dataset['Period'] = trade_dataset['Period'].astype(int)

# Additional edge characteristics
cepii_edge_dataset = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/cepii_edge.csv')
# Drop countries that cannot be mapped
cepii_edge_dataset = cepii_edge_dataset[(cepii_edge_dataset['iso_o']!='ANT') & (cepii_edge_dataset['iso_d']!='ANT')
                                        & (cepii_edge_dataset['iso_o']!='PAL') & (cepii_edge_dataset['iso_d']!='PAL')
                                        & (cepii_edge_dataset['iso_o']!='TMP') & (cepii_edge_dataset['iso_d']!='TMP')
                                        & (cepii_edge_dataset['iso_o']!='YUG') & (cepii_edge_dataset['iso_d']!='YUG')
                                        & (cepii_edge_dataset['iso_o']!='ZAR') & (cepii_edge_dataset['iso_d']!='ZAR')]
cepii_edge_dataset['iso_o'] = cepii_edge_dataset['iso_o'].apply(lambda x: x if x!='ROM' else 'ROU')
cepii_edge_dataset['iso_d'] = cepii_edge_dataset['iso_d'].apply(lambda x: x if x!='ROM' else 'ROU')

cepii_edge_dataset['iso_o'] = cepii_edge_dataset['iso_o'].astype('string').apply(lambda x: iso_alpha3_numeric_mapper[x])
cepii_edge_dataset['iso_d'] = cepii_edge_dataset['iso_d'].astype('string').apply(lambda x: iso_alpha3_numeric_mapper[x])

# Node characteristics
wbg_dataset = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/WBG_data_all_countries.csv')
wbg_dataset['economy'] = wbg_dataset['economy'].astype('string').apply(lambda x: iso_alpha3_numeric_mapper[x])
wbg_dataset.columns = ['economy', 'year', 'gdp', 'total_population', 'urban_population(%_of_total)']
wbg_dataset['year'] = wbg_dataset['year'].apply(lambda x: x[2:])
wbg_dataset['year'] = wbg_dataset['year'].astype(int)

country_groups_dataset = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/country_groups.csv')
# Drop countries that cannot be mapped
country_groups_dataset = country_groups_dataset[(country_groups_dataset['country_code']!='JA')]
country_groups_dataset['country_code'] = country_groups_dataset['country_code'].apply(lambda x: iso_alpha2_numeric_mapper[x])

cepii_nodes_dataset = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/cepii_node.csv')
# Drop countries that cannot be mapped
cepii_nodes_dataset = cepii_nodes_dataset[(cepii_nodes_dataset['iso3']!='ANT')
                                        & (cepii_nodes_dataset['iso3']!='PAL') 
                                        & (cepii_nodes_dataset['iso3']!='TMP')
                                        & (cepii_nodes_dataset['iso3']!='YUG')
                                        & (cepii_nodes_dataset['iso3']!='ZAR')
                                        ]
cepii_nodes_dataset['iso3'] = cepii_nodes_dataset['iso3'].apply(lambda x: x if x!='ROM' else 'ROU')
cepii_nodes_dataset['iso_3'] = cepii_nodes_dataset['iso3']
cepii_nodes_dataset['iso3'] = cepii_nodes_dataset['iso3'].apply(lambda x: iso_alpha3_numeric_mapper[x])

# Drop city info as it causes duplicates
cepii_nodes_dataset.drop(columns=["city_en", "city_fr", "lat", "lon", "cap", "maincity"], inplace=True)
cepii_nodes_dataset.drop_duplicates(inplace=True)

###############
# Validate trade data
###############

# Obtain list of countries from trade dataset and validate

reporter_set = set(trade_dataset.Reporter.unique())
partner_set = set(trade_dataset.Partner.unique())

if (reporter_set - partner_set != set()) & (partner_set - reporter_set != set()):
    print('Number of partners and reporters do no match!')

periods = set(trade_dataset.Period.unique())
all_pairs = set([(i,j,k) for i in reporter_set for j in partner_set for k in periods])
real_pairs = set(list(trade_dataset[['Reporter', 'Partner', 'Period']].itertuples(index=False, name=None)))

if (all_pairs - real_pairs != set()) & (real_pairs - all_pairs != set()):
    print('Number of expected and real observations do no match!')

# Clean errorous codes

iso_codes_in_country_names_in_trade_dataset = set(country_names_in_trade_dataset['ISO_3166-1_numeric_code'])
if (reporter_set - iso_codes_in_country_names_in_trade_dataset != set()) & (iso_codes_in_country_names_in_trade_dataset - reporter_set != set()):
    print('ISO codes in trade dataset and country_names_in_trade_dataset do not match!')

errorous_country_code_mapper = dict(zip(reporter_set, reporter_set))
errorous_country_code_mapper[251] = 250
errorous_country_code_mapper[579] = 578
errorous_country_code_mapper[699] = 356
errorous_country_code_mapper[757] = 756
errorous_country_code_mapper[842] = 840

trade_dataset['Reporter'] = trade_dataset['Reporter'].apply(lambda x: errorous_country_code_mapper[x])
trade_dataset['Partner'] = trade_dataset['Partner'].apply(lambda x: errorous_country_code_mapper[x])

###############
# Merge cepii to trade data
###############

trade_dataset['Reporter'] = trade_dataset['Reporter'].astype('float64')
trade_dataset['Partner'] = trade_dataset['Partner'].astype('float64')

trade_edgelist = pd.merge(trade_dataset, cepii_edge_dataset, left_on = ['Reporter', 'Partner'], right_on = ['iso_o','iso_d'], how='left')
trade_edgelist['iso_o'] = trade_edgelist['iso_o'].astype('int')
trade_edgelist['iso_d'] = trade_edgelist['iso_d'].astype('int')
trade_edgelist.drop(columns=['Reporter', 'Partner'], inplace=True)

###############
# Drop edges where origin = destination
###############

trade_edgelist = trade_edgelist[trade_edgelist['iso_o'] != trade_edgelist['iso_d']]

###############
# Filter wbg to trade data countries
# Merge with cepii_nodes and country groups
###############

trade_dataset_countries = trade_dataset['Reporter'].unique()
wbg_dataset = wbg_dataset[wbg_dataset['economy'].isin(trade_dataset_countries)]

#trade_nodelist = pd.merge(wbg_dataset, country_groups_dataset, left_on = ['economy'], right_on = ['country_code'], how='left')
trade_nodelist = pd.merge(wbg_dataset, cepii_nodes_dataset, left_on = ['economy'], right_on = ['iso3'], how='left')
trade_nodelist['iso_numeric'] = trade_nodelist['iso3'].astype('int')
trade_nodelist.drop(columns=['economy', 'iso3'], inplace=True)

###############
# Use backfill for gdp data
###############
trade_nodelist.gdp = trade_nodelist.groupby('country').gdp.bfill()

###############
# Fill citynum for Macao
###############
trade_nodelist['citynum'] = trade_nodelist.apply(lambda x: 1 if x['iso_numeric']==446 else x['citynum'], axis=1)

###############
# Fill distw,distwces for Macao
###############
trade_edgelist['distw'] = trade_edgelist.apply(lambda x: x['dist'] if (x['iso_o']==446) | (x['iso_d']==446) else x['distw'], axis=1)
trade_edgelist['distwces'] = trade_edgelist.apply(lambda x: x['dist'] if (x['iso_o']==446) | (x['iso_d']==446) else x['distwces'], axis=1)

###############
# Add reverse flow value
###############

trade_edgelist_reverse = trade_edgelist[["Period", "Value", "iso_o", "iso_d"]].copy()
trade_edgelist_reverse.rename(columns = {"Value": "Value_reverse", "iso_o":"iso_d", "iso_d":"iso_o"}, inplace=True)

trade_edgelist = pd.merge(trade_edgelist, trade_edgelist_reverse, on=["Period", "iso_o", "iso_d"], how="left")

###############
# All to node
###############
all_to_node = trade_edgelist.groupby(["Period",  "iso_d"])["Value"].sum().reset_index()

all_to_node.rename(columns={"Value": "all_to_d"}, inplace=True)

trade_edgelist = pd.merge(trade_edgelist, all_to_node, on=["Period", "iso_d"], how="left")

all_to_node.rename(columns={"all_to_d": "all_to_o", "iso_d":"iso_o"}, inplace=True)

trade_edgelist = pd.merge(trade_edgelist, all_to_node, on=["Period", "iso_o"], how="left")

###############
# Node to all
###############
node_to_all = trade_edgelist.groupby(["Period",  "iso_o"])["Value"].sum().reset_index()

node_to_all.rename(columns={"Value": "o_to_all"}, inplace=True)

trade_edgelist = pd.merge(trade_edgelist, node_to_all, on=["Period", "iso_o"], how="left")

node_to_all.rename(columns={"o_to_all": "d_to_all", "iso_o":"iso_d"}, inplace=True)

trade_edgelist = pd.merge(trade_edgelist, node_to_all, on=["Period", "iso_d"], how="left")

###############
# Save datasets
###############
trade_nodelist.to_csv('../Output_datasets/Yearly_trade_data_prediction/trade_nodelist.csv')
trade_edgelist.to_csv('../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv')

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

# Chunk parameters
chunk_size = 5

# Chunked data path
chunk_path  = "../Output_datasets/Yearly_trade_data_prediction/Chunked_merged_data"

# Filter only neccesary columns
nodes_columns = node_features + [node_id] + [node_timestamp]
trade_nodelist = trade_nodelist[nodes_columns]

edges_columns = flows_features + [flow_origin] + [flow_destination] + \
    [flows_timestamp] + [flows_value]
trade_edgelist = trade_edgelist[edges_columns]

# Merge nodes and edges
trade_nodelist.rename(columns={node_id: flow_origin, node_timestamp: flows_timestamp}, inplace=True)
trade_nodelist[flow_destination] = trade_nodelist[flow_origin]

nodes_and_edges = pd.merge(pd.merge(trade_edgelist, trade_nodelist.drop(flow_destination, axis=1), how='left', on=[flow_origin, flows_timestamp]),
                        trade_nodelist.drop(flow_origin, axis=1), how='left', on=[flow_destination, flows_timestamp], 
                        suffixes=('_o', '_d'))

chunk_period_list = [nodes_and_edges[flows_timestamp].unique()[i:i+chunk_size] for i in range(0, len(nodes_and_edges[flows_timestamp].unique())-(chunk_size-1))]
for chunk in chunk_period_list[:-1]:
    nodes_and_edges_chunk = nodes_and_edges[nodes_and_edges[flows_timestamp].isin(chunk)]
    chunk_name = str(min(chunk)) + "-" + str(max(chunk))
    nodes_and_edges_chunk.to_csv(f"{chunk_path}/{chunk_name}.csv", index=False)