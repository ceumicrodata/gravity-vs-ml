import torch.utils.data.distributed


########################
# Function for training
########################

def train(dataloader, model, optimizer, loss_fn = None):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        if loss_fn==None:
            loss = model.loss(pred, y)
        else:
            loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, flow_data, loss_fn = None, store_predictions = False):
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    model.eval()
    test_loss = 0
    with torch.no_grad():
        batch = 0
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