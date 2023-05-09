import sys
import os

import torch.utils.data.distributed
import torch.optim as optim
from ray import tune
import numpy as np

from deepgravity import DeepGravity


########################
# Function for training
########################

#def train(dataloader, model, optimizer, loss_fn = None):
#    size = len(dataloader.dataset)
#    model.train()
#    for batch, (X, y) in enumerate(dataloader):
#        # Compute prediction error
#        pred = model(X)
#        if loss_fn==None:
#            loss = model.loss(pred, y)
#        else:
#            loss = loss_fn(pred, y)
#
#        # Backpropagation
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        if batch % 10 == 0:
#            loss, current = loss.item(), (batch + 1) * len(X)
#            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(dataloader, model, optimizer, epoch, loss_fn = None):
    running_loss = 0.0
    epoch_steps = 0
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
        running_loss += loss.item()
        epoch_steps += 1
        if batch % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, batch + 1,
                                            running_loss / epoch_steps))
            running_loss = 0.0

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
    return val_loss

def train_and_validate_deepgravity(config,
                             train_data_chunked,
                             validation_data_chunked,
                             chunk,
                             momentum,
                             #epochs,
                             loss_fn = None,
                             checkpoint_dir=None):

    # Load data
    train_data_loader = torch.utils.data.DataLoader(train_data_chunked[chunk], batch_size=config["batch_size"])
    validation_data_loader = torch.utils.data.DataLoader(validation_data_chunked[chunk], batch_size=config["batch_size"])

    input_dim = train_data_chunked[chunk].get_feature_dim()

    deep_gravity_model = DeepGravity(dim_input = input_dim,
                                    dim_hidden = config["dim_hidden"],
                                    dropout_p = config["dropout_p"],
                                    num_layers = config["num_layers"],)

    optimizer = optim.RMSprop(deep_gravity_model.parameters(), lr=config["lr"], momentum=momentum)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join("checkpoints", str(chunk), checkpoint_dir, "checkpoint"))
        deep_gravity_model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    #val_losses = [np.exp(100), np.exp(100), np.exp(100)]

    early_stopper = EarlyStopper(patience=10, min_delta=3e+8) #trade: min_delta=3e+17

    for epoch in range(config["epochs"]):
        #print(f"Epoch {epoch+1}\n-------------------------------")
        train_model(train_data_loader, deep_gravity_model, optimizer, epoch, loss_fn)
        val_loss = validate_model(validation_data_loader, deep_gravity_model,
                                                loss_fn)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((deep_gravity_model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss)

        if epoch>100:
            if early_stopper.early_stop(val_loss):
                break

    print("Finished training!")


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
                flow_data_keys = list(flow_data.data_dict.keys())[batch_size*batch:batch_size*batch+batch_size]
                for element in range(len(flow_data_keys)):
                    flow_data.data_dict[flow_data_keys[element]]['prediction'] = pred[element].numpy()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
