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
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

import trainer.model_utils as model_utils
from trainer.data_compiler import FlowDataset
from trainer.neural_network import DeepGravityLinOutput, DeepGravityReluOutput

# Load data
if sys.argv[1]=="GeoDS_full":
    import trainer.parameters_GeoDS_full as parameters
elif sys.argv[1]=="GeoDS_limited":
    import trainer.parameters_limited_GeoDS as parameters
elif sys.argv[1]=="GeoDS_limited_lag":
    import trainer.parameters_limited_lag_GeoDS as parameters
elif sys.argv[1]=="Google":
    import trainer.parameters_Google_full as parameters
elif sys.argv[1]=="Google_limited":
    import trainer.parameters_limited_Google as parameters
elif sys.argv[1]=="Google_limited_lag":
    import trainer.parameters_limited_lag_Google as parameters
elif sys.argv[1]=="trade":
    import trainer.parameters_trade_full as parameters
elif sys.argv[1]=="trade_limited":
    import trainer.parameters_limited_trade as parameters
elif sys.argv[1]=="trade_limited_lag":
    import trainer.parameters_limited_lag_trade as parameters

# random seeds
random.seed(parameters.seed)     # python random generator
np.random.seed(parameters.seed)  # numpy random generator

torch.manual_seed(parameters.seed)
torch.cuda.manual_seed_all(parameters.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch_device = torch.device("cpu")

torch.use_deterministic_algorithms(True)

columns = {'node_id': parameters.node_id,
           'node_timestamp': parameters.node_timestamp,
           'node_features': parameters.node_features,
           'flow_origin': parameters.flow_origin,
           'flow_destination': parameters.flow_destination,
           'flows_timestamp': parameters.flows_timestamp,
           'target_value':parameters.flows_value,
           'flows_features':parameters.flows_features}

##############
# Load data
##############

all_files = glob.glob(os.path.join(parameters.chunk_path , "*.csv"))

train_data_chunked = []
validation_data_chunked = []
train_validation_data_chunked = []
test_data_chunked = []

for chunk_file in tqdm.tqdm(all_files):
    nodes_edges = pd.read_csv(chunk_file, index_col=None, header=0, thousands=',')
    nodes_edges.drop(['date','start_date'], errors='ignore', axis=1, inplace=True)

    ##############
    # Filter columns
    ##############
    filter_columns = []
    filter_columns.extend(parameters.node_features + [parameters.node_id] + [parameters.node_timestamp])
    filter_columns.extend([i+'_o' for i in parameters.node_features])
    filter_columns.extend([i+'_d' for i in parameters.node_features])
    filter_columns.extend([parameters.flow_origin] + [parameters.flow_destination] + [parameters.flows_timestamp])
    filter_columns.extend([parameters.flows_value] + parameters.flows_features)
    
    nodes_edges = nodes_edges.filter(set(filter_columns))
    
    ##############
    # Create data objects
    ##############
    flow_chunk = FlowDataset(domain=parameters.domain,
                            columns=columns,
                            nodes_edges=nodes_edges)

    flow_chunk.add_past_values(periods=parameters.lag_periods,
                            edge_columns = parameters.time_dependent_edge_columns,
                            node_columns = parameters.time_dependent_node_columns)

    flow_chunk.add_target_values()

    # Create a list of FlowDataset objects
    train_data, validation_data, test_data = flow_chunk.split_train_validate_test_scale(
        validation_period = parameters.validation_period,
        columns_to_scale=parameters.columns_to_scale,
        custom_seed = parameters.seed
    )
    train_data_chunked.append(train_data)
    validation_data_chunked.append(validation_data)
    test_data_chunked.append(test_data)

###################
# Run DeepGravity
###################

prediction_list = []
best_config_list = []
for chunk in range(len(train_data_chunked)):
    # Set scheduler
    scheduler = HyperBandScheduler( #ASHAScheduler( # Not deterministic 
        metric="loss",
        mode="min",
        max_t=parameters.max_epochs,
        reduction_factor=parameters.reduction_factor)
    # Set reporter
    reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"])

    config = parameters.config

    config["domain"] = parameters.domain
    config["chunk"] = chunk
    config["momentum"] = parameters.momentum
    config["weight_decay"] = parameters.weight_decay
    config["early_stopper_patience"] = parameters.early_stopper_patience
    config["early_stopper_min_delta"] = parameters.early_stopper_min_delta
    config["early_stopper_grace_period"] = parameters.early_stopper_grace_period
    config["loss_fn"] = parameters.loss_fn
    config["seed"] = parameters.seed

    # Run tuning
    result = tune.run(
        tune.with_parameters(model_utils.train_and_validate_deepgravity, train_data_chunks=train_data_chunked,
        validation_data_chunks=validation_data_chunked),
        resources_per_trial={"cpu": parameters.resources_per_trial},
        config=config,
        num_samples=parameters.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        reuse_actors=False,
        )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    best_config_list.append(pd.DataFrame.from_dict([best_trial.config]))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    input_dim = train_data_chunked[chunk].get_feature_dim()

    if parameters.domain=='Google':
        best_trained_model = DeepGravityLinOutput(dim_input = input_dim,
                                        dim_hidden = best_trial.config["dim_hidden"],
                                        dropout_p = best_trial.config["dropout_p"],
                                        num_layers = best_trial.config["num_layers"],)
    else:
        best_trained_model = DeepGravityReluOutput(dim_input = input_dim,
                                        dim_hidden = best_trial.config["dim_hidden"],
                                        dropout_p = best_trial.config["dropout_p"],
                                        num_layers = best_trial.config["num_layers"],) 

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    best_checkpoint_dir = best_checkpoint.to_directory(path=os.path.join(parameters.output_path, "best_checkpoints", parameters.domain, str(chunk), f"checkpoint_{str(datetime.datetime.now()).replace(' ', '_')[:19]}"))
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    test_data_loader = torch.utils.data.DataLoader(test_data_chunked[chunk],
                                                   batch_size=best_trial.config["batch_size"],
                                                   worker_init_fn=seed_worker,
                                                   generator=g,
                                                   shuffle=False)
    model_utils.test(test_data_loader, best_trained_model, test_data_chunked[chunk], loss_fn = parameters.loss_fn, store_predictions=True)
    print("Finished prediction on test set")
    prediction_list.append(test_data_chunked[chunk].compile_predictions(columns_to_rename = parameters.columns_to_rename))

###################
# Save predictions
###################

pd.concat(prediction_list, axis=0).to_csv(f"{parameters.output_path}/prediction_{str(datetime.datetime.now()).replace(' ', '_')[:19]}.csv")
pd.concat(best_config_list, axis=0).to_csv(f"{parameters.output_path}/best_configs_{str(datetime.datetime.now()).replace(' ', '_')[:19]}.csv")

