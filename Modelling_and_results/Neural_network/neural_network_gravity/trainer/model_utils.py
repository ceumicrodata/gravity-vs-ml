import sys
import os

import torch.utils.data.distributed
import torch.optim as optim
import torch
from ray import tune
import numpy as np
import random

from trainer.deepgravity import DeepGravityLinOutput, DeepGravityReluOutput

########################
# Function for training
########################

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss * 1.025): #+ self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(dataloader, model, optimizer, epoch, loss_fn = None):
    num_batches = len(dataloader)
    train_loss = 0.0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        pred = model(X)
        if loss_fn==None:
            loss = model.loss(pred, y)
        else:
            loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        # print statistics
        #train_loss += loss.item()
    #train_loss /= num_batches
    #print(train_loss)


def validate_model(dataloader, model, loss_fn = None):
    num_batches = len(dataloader)
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            pred = model(X)
            if loss_fn==None:
                loss = model.loss(pred, y)
            else:
                loss = loss_fn(pred, y)
            val_loss += loss.numpy()

    val_loss /= num_batches
    #print("Validation loss")
    #print(val_loss)
    return val_loss

def test(dataloader, model, flow_data, loss_fn = None, store_predictions = False):
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            if loss_fn==None:
                test_loss += model.loss(pred, y).item()
            else:
                test_loss += loss_fn(pred, y).item()

            if store_predictions:
                flow_data_keys = list(flow_data.nodes_edges.index)[batch_size*batch:batch_size*batch+batch_size]
                for element in range(len(flow_data_keys)):
                    flow_data.nodes_edges.loc[flow_data_keys[element], 'prediction'] = pred[element].numpy()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def train_and_validate_deepgravity(config,
                             train_data_chunks,
                             validation_data_chunks,
                             loss_fn = None,
                             checkpoint_dir=None):

    domain = config["domain"]
    chunk = config["chunk"]
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]
    early_stopper_patience = config["early_stopper_patience"]
    early_stopper_min_delta = config["early_stopper_min_delta"]
    early_stopper_grace_period = config["early_stopper_grace_period"]
    loss_fn = config["loss_fn"]
    seed = config["seed"]

    # random seeds
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch_device = torch.device("cpu")

    torch.use_deterministic_algorithms(True)

    g = torch.Generator()
    g.manual_seed(0)

    train_data_loader = torch.utils.data.DataLoader(train_data_chunks[chunk],
                                                    batch_size=config["batch_size"],
                                                    worker_init_fn=seed_worker,
                                                    generator=g,
                                                    shuffle=False)
    validation_data_loader = torch.utils.data.DataLoader(validation_data_chunks[chunk],
                                                        batch_size=config["batch_size"],
                                                        worker_init_fn=seed_worker,
                                                        generator=g,
                                                        shuffle=False)

    input_dim = train_data_chunks[chunk].get_feature_dim()

    if domain=="Google":
        deep_gravity_model = DeepGravityLinOutput(dim_input = input_dim,
                                        dim_hidden = config["dim_hidden"],
                                        dropout_p = config["dropout_p"],
                                        num_layers = config["num_layers"],)
    else:
        deep_gravity_model = DeepGravityReluOutput(dim_input = input_dim,
                                        dim_hidden = config["dim_hidden"],
                                        dropout_p = config["dropout_p"],
                                        num_layers = config["num_layers"],)

    optimizer = optim.Adam(deep_gravity_model.parameters(),
                                lr=config["lr"],
                                weight_decay=weight_decay)

    #optimizer = optim.RMSprop(deep_gravity_model.parameters(), lr=config["lr"], momentum=momentum,
    #weight_decay=weight_decay)

    early_stopper = EarlyStopper(patience=early_stopper_patience,
                                min_delta=early_stopper_min_delta)

    for epoch in range(config["epochs"]):
        train_model(train_data_loader, deep_gravity_model, optimizer, epoch, loss_fn)
        val_loss = validate_model(validation_data_loader, deep_gravity_model,
                                                loss_fn)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((deep_gravity_model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss)

        if epoch>early_stopper_grace_period:
            if early_stopper.early_stop(val_loss):
                break

    print("Finished training!")

