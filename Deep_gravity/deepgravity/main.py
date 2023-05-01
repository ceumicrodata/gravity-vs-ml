import pandas as pd
import numpy as np
import os
import datetime
import sys
import tqdm

import random
import torch.utils.data.distributed
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import model_utils
from data_compiler import FlowDataset
from deepgravity import DeepGravity

sys.path.append('../trade_predictions/')
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
if parameters.domain=='Google':
    node_targets = pd.read_csv(parameters.node_target_path)
    nodes = pd.merge(nodes, node_targets, on=[parameters.node_id], how='inner')

nodes_columns = parameters.node_features + parameters.node_targets + [parameters.node_id] + [parameters.node_timestamp]
nodes_columns = [i for i in nodes_columns if i != ""]
nodes_columns = nodes_columns
nodes = nodes[nodes_columns]

if parameters.domain=='Google':
    # Rename and melt
    nodes.rename(columns={parameters.node_id: parameters.flow_origin}, inplace=True)

    nodes = pd.melt(nodes, id_vars=parameters.node_features+[parameters.flow_origin]+[parameters.node_timestamp],
                            value_vars=parameters.node_targets, var_name='destination', value_name='Value')

edges = pd.read_csv(parameters.edge_path)
if parameters.domain=="GeoDS":
    edge_targets = pd.read_csv(parameters.edge_target_path)
    edges = pd.merge(edges, edge_targets, on=[parameters.flow_origin, parameters.flow_destination], how='inner')

edges_columns = parameters.flows_features + [parameters.flow_origin] + \
    [parameters.flow_destination] + [parameters.flows_timestamp] + \
        [parameters.flows_value]
edges_columns = [i for i in edges_columns if i != ""]
edges = edges[edges_columns]

if parameters.domain=='Google':
    unit = [parameters.flow_origin, parameters.node_timestamp]
    target_value = 'Value'
else:
    unit = [parameters.flow_origin, parameters.flows_timestamp]
    target_value = parameters.flows_value

##############
# Initial cleaning
##############

nodes = nodes.fillna(0)
edges = edges.fillna(0)

##############
# Create data objects
##############

columns = {'node_id': parameters.node_id,
           'node_timestamp': parameters.node_timestamp,
           'flow_origin': parameters.flow_origin,
           'flow_destination': parameters.flow_destination,
           'flows_timestamp': parameters.flows_timestamp,
           'target_value':target_value}

flow_data = FlowDataset(domain=parameters.domain,
                        columns=columns,
                        unit = unit,
                        nodes=nodes,
                        edges=edges,)

# Create a list of FlowDataset objects
flow_data_chunked = flow_data.create_chunks(chunk_size=parameters.chunk_size, window_size=parameters.window_size)

# Add past values to each chunk
[flow_chunk.add_past_values(periods=parameters.lag_periods,
                            edge_columns = parameters.time_dependent_edge_columns,
                            node_columns = parameters.time_dependent_node_columns) for flow_chunk in tqdm.tqdm(flow_data_chunked)]

# Add target to each chunk
[flow_chunk.add_target_values() for flow_chunk in tqdm.tqdm(flow_data_chunked)]

# Create a list of FlowDataset objects
train_data_chunked = []
validation_data_chunked = []
test_data_chunked = []

for flow_data in tqdm.tqdm(flow_data_chunked):
    train_data, validation_data, test_data = flow_data.split_train_validate_test(validation_period = parameters.validation_period)
    train_data_chunked.append(train_data)
    validation_data_chunked.append(validation_data)
    test_data_chunked.append(test_data)

###################
# Run DeepGravity
###################

prediction_list = []
for chunk in range(len(train_data_chunked)):
    # Set scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=parameters.epochs,
        grace_period=1,
        reduction_factor=2)
    # Set reporter
    reporter = CLIReporter(
            # parameter_columns=["lr", "batch_size", "dim_hidden", "dropout_p", "num_layers"],
            metric_columns=["loss", "training_iteration"])
    # Run tuning
    result = tune.run(
        tune.with_parameters(model_utils.train_and_validate_deepgravity, train_data_chunked = train_data_chunked,
                validation_data_chunked = validation_data_chunked, chunk = chunk, momentum = parameters.momentum,
                epochs = parameters.epochs,loss_fn = parameters.loss_fn),
        resources_per_trial={"cpu": 4},
        config=parameters.config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    input_dim = train_data_chunked[chunk].get_feature_dim()
    best_trained_model = DeepGravity(dim_input = input_dim,
                                    dim_hidden = best_trial.config["dim_hidden"],
                                    dropout_p = best_trial.config["dropout_p"],
                                    num_layers = best_trial.config["num_layers"],)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    best_checkpoint_dir = best_checkpoint.to_directory(path=os.path.join(parameters.output_path, "best_checkpoints", "trade", str(chunk), f"checkpoint_{str(datetime.datetime.now()).replace(' ', '_')[:19]}"))
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_data_loader = torch.utils.data.DataLoader(test_data_chunked[chunk], batch_size=4)
    model_utils.test(test_data_loader, best_trained_model, test_data_chunked[chunk], loss_fn = parameters.loss_fn, store_predictions=True)
    print("Finished prediction on test set")
    prediction_list.append(test_data_chunked[chunk].compile_predictions(columns_to_rename = parameters.columns_to_rename))

###################
# Save predictions
###################

pd.concat(prediction_list, axis=0).to_csv(f"{parameters.output_path}/prediction_{str(datetime.datetime.now()).replace(' ', '_')[:19]}.csv")
