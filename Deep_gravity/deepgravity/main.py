import pandas as pd
import numpy as np
import datetime
import sys
import tqdm

import random
import torch.optim as optim
import torch.utils.data.distributed
from torch import nn

import model_utils
from data_compiler import FlowDataset
from deepgravity import DeepGravity

sys.path.append('../trade_predictions')
import parameters

# random seeds
torch.manual_seed(parameters.seed)
np.random.seed(parameters.seed)
random.seed(parameters.seed)

torch_device = torch.device("cpu")

##############
# Load data
##############

nodes = pd.read_csv(parameters.node_path)
nodes_columns = parameters.node_features + [parameters.node_id] + [parameters.node_timestamp]
nodes = nodes[nodes_columns]

edges = pd.read_csv(parameters.edge_path)
edges_columns = parameters.flows_features + [parameters.flow_origin] + \
    [parameters.flow_destination] + [parameters.flows_timestamp] + \
        [parameters.flows_value]
edges = edges[edges_columns]

##############
# Initial cleaning
##############

nodes = nodes.fillna(0)

##############
# Create data objects
##############

columns = {'node_id': parameters.node_id,
           'node_timestamp': parameters.node_timestamp,
           'flow_origin': parameters.flow_origin,
           'flow_destination': parameters.flow_destination,
           'flows_timestamp': parameters.flows_timestamp,
           'flows_value':parameters.flows_value}

flow_data = FlowDataset(columns=columns,
                        unit = [parameters.flow_origin, parameters.flows_timestamp],
                        nodes=nodes,
                        edges=edges,)

# Create a list of FlowDataset objects
flow_data_chunked = flow_data.create_chunks(chunk_size=6)

# Add past values to each chunk
[flow_chunk.add_past_values(periods=parameters.lag_periods,
                            edge_columns = parameters.time_dependent_edge_columns,
                            node_columns = parameters.time_dependent_node_columns) for flow_chunk in tqdm.tqdm(flow_data_chunked)]

# Add target to each chunk
[flow_chunk.add_target_values() for flow_chunk in tqdm.tqdm(flow_data_chunked)]

# Create a list of FlowDataset objects
train_data_chunked = []
test_data_chunked = []

for flow_data in tqdm.tqdm(flow_data_chunked):
    train_data, test_data = flow_data.split_train_test(test_period = 1)
    train_data_chunked.append(train_data)
    test_data_chunked.append(test_data)

###################
# Run DeepGravity
###################

prediction_list = []
for chunk in range(len(train_data_chunked)):
    train_data_loader = torch.utils.data.DataLoader(train_data_chunked[chunk], batch_size=parameters.batch_size)
    test_data_loader = torch.utils.data.DataLoader(test_data_chunked[chunk], batch_size=parameters.batch_size)

    input_dim = train_data_chunked[chunk].get_feature_dim()

    deep_gravity_model = DeepGravity(dim_input = input_dim,
                                    dim_hidden = parameters.dim_hidden)
    
    optimizer = optim.RMSprop(deep_gravity_model.parameters(), lr=parameters.lr, momentum=parameters.momentum)

    for t in range(parameters.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model_utils.train(train_data_loader, deep_gravity_model, optimizer)
        model_utils.test(test_data_loader, deep_gravity_model, test_data_chunked[chunk], loss_fn = None)

    model_utils.test(test_data_loader, deep_gravity_model, test_data_chunked[chunk], loss_fn = None, store_predictions=True)
    prediction_list.append(test_data_chunked[chunk].compile_predictions(columns_to_rename = parameters.columns_to_rename))
    print("Done!")

print("Training and prediction was successful!")

###################
# Save predictions
###################

pd.concat(prediction_list, axis=0).to_csv(f"{parameters.output_path}/prediction_{str(datetime.datetime.now()).replace(' ', '_')[:19]}.csv")



