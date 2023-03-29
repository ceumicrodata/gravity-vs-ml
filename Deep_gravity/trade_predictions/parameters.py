##############
# Dataset params
##############
domain='trade'
node_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_nodelist.csv'
edge_path = '../../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv'
output_path = '../trade_predictions/'

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
                'comcol', 'curcol', 'col45', 'smctry', 'dist', 'distcap']

# Chunk parameters
chunk_size = 6
train_periods = 5

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
# Model params
##############
batch_size=10
epochs=25
lr=5e-6
momentum=0.9
dim_hidden = 64
seed=1234
device='cpu'

# TBA add mode to only evaluate
#mode='train'