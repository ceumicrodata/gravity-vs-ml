import pandas as pd
import numpy as np
import os
import datetime
import sys
import tqdm
import glob

import random
import torch.utils.data.distributed
import torch.optim as optim
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import model_utils
from data_compiler import FlowDataset
from deepgravity import DeepGravity

if sys.argv[1]=="GeoDS":
    sys.path.append('../GeoDS_mobility_predictions/')
    import parameters
elif sys.argv[1]=="Google":
    sys.path.append('../google_mobility_predictions/')
    import parameters
elif sys.argv[1]=="trade":
    sys.path.append('../trade_predictions/')
    import parameters


# random seeds
torch.manual_seed(parameters.seed)
np.random.seed(parameters.seed)
random.seed(parameters.seed)

torch_device = torch.device("cpu")

if parameters.domain=='Google':
    unit = ["index_aux", parameters.node_timestamp] #parameters.flow_origin
    target_value = parameters.node_target
else:
    unit = [parameters.flow_origin, parameters.flows_timestamp]
    target_value = parameters.flows_value

columns = {'node_id': parameters.node_id,
           'node_timestamp': parameters.node_timestamp,
           'flow_origin': parameters.flow_origin,
           'flow_destination': parameters.flow_destination,
           'flows_timestamp': parameters.flows_timestamp,
           'target_value':target_value}

train_data_chunked = []
validation_data_chunked = []
train_validation_data_chunked = []
test_data_chunked = []

##############
# Load data
##############

all_files = glob.glob(os.path.join(parameters.chunk_path , "*.csv"))

for chunk_file in tqdm.tqdm(all_files):
    nodes_edges = pd.read_csv(chunk_file, index_col=None, header=0, thousands=',')
    nodes_edges.drop(['date','start_date'], errors='ignore', axis=1, inplace=True)

    if parameters.domain=='Google':
        nodes_edges["index_aux"] = "0"
    ##############
    # Create data objects
    ##############
    flow_chunk = FlowDataset(domain=parameters.domain,
                            columns=columns,
                            unit = unit,
                            nodes_edges=nodes_edges)

    flow_chunk.add_past_values(periods=parameters.lag_periods,
                            edge_columns = parameters.time_dependent_edge_columns,
                            node_columns = parameters.time_dependent_node_columns)

    flow_chunk.add_target_values()

    # Create a list of FlowDataset objects
    train_data, validation_data, train_validation_data, test_data = flow_chunk.split_train_validate_test_logs(
        validation_period = parameters.validation_period,
        columns_to_log=parameters.columns_to_log
    )
    train_data_chunked.append(train_data)
    validation_data_chunked.append(validation_data)
    train_validation_data_chunked.append(train_validation_data)
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
        max_t=parameters.max_epochs,
        grace_period=parameters.grace_period,
        reduction_factor=parameters.reduction_factor)
    # Set reporter
    reporter = CLIReporter(
            # parameter_columns=["lr", "batch_size", "dim_hidden", "dropout_p", "num_layers"],
            metric_columns=["loss", "training_iteration"])
    # Run tuning
    result = tune.run(
        tune.with_parameters(model_utils.train_and_validate_deepgravity, train_data_chunked = train_data_chunked,
                validation_data_chunked = validation_data_chunked, chunk = chunk, momentum = parameters.momentum,
                weight_decay=parameters.weight_decay, early_stopper_patience = parameters.early_stopper_patience,
                early_stopper_min_delta = parameters.early_stopper_min_delta,
                early_stopper_grace_period = parameters.early_stopper_grace_period,
                loss_fn = parameters.loss_fn),
        resources_per_trial={"cpu": parameters.resources_per_trial},
        config=parameters.config,
        num_samples=parameters.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        reuse_actors=False)

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
    best_checkpoint_dir = best_checkpoint.to_directory(path=os.path.join(parameters.output_path, "best_checkpoints", parameters.domain, str(chunk), f"checkpoint_{str(datetime.datetime.now()).replace(' ', '_')[:19]}"))
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    #train_validation_data_loader = torch.utils.data.DataLoader(train_validation_data_chunked[chunk], batch_size=32)
    #optimizer = optim.RMSprop(best_trained_model.parameters(), lr=best_trial.config["lr"], momentum=parameters.momentum)
    #for epoch in range(best_trial.config["epochs"]):
    #    #print(f"Epoch {epoch+1}\n-------------------------------")
    #    model_utils.train_model(train_validation_data_loader, best_trained_model, optimizer, epoch, parameters.loss_fn)

    test_data_loader = torch.utils.data.DataLoader(test_data_chunked[chunk], batch_size=parameters.data_loader_batch_size)
    model_utils.test(test_data_loader, best_trained_model, test_data_chunked[chunk], loss_fn = parameters.loss_fn, store_predictions=True)
    print("Finished prediction on test set")
    prediction_list.append(test_data_chunked[chunk].compile_predictions(columns_to_rename = parameters.columns_to_rename))

###################
# Save predictions
###################

pd.concat(prediction_list, axis=0).to_csv(f"{parameters.output_path}/prediction_{str(datetime.datetime.now()).replace(' ', '_')[:19]}.csv")
