import pickle
import os

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import (
    temporal_signal_split,
    StaticGraphTemporalSignal,
)


class RecurrentGCN(torch.nn.Module):
    """Class for a pytorch neural network module"""

    def __init__(self, node_features: int, out_channels: int, filter_size: int):
        """
        Initialize a pytorch model with DCRNN architecture

        Args:
            node_features:
            Number of node features to use.
            out_channels:
            Number of DCRNN hidden features.
            filter_size:
            DCRNN filter size.
        """
        super().__init__()
        self.recurrent = DCRNN(node_features, out_channels, filter_size)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """Perform a forward feed
        Args:
            x:
            feature Pytorch Float Tensor
            edge_index:
            Pytorch Float Tensor of edge indices
            edge_weight:
            Pytorch Float Tensor of edge weights

        Returns:
            tens:
            Pytorch Float Tensor of Hidden state matrix for all nodes
        """
        tens = self.recurrent(x, edge_index, edge_weight)
        tens = F.relu(tens)
        tens = self.linear(tens)
        return tens


def train_full(config, checkpoint_dir=None, data_dir=None):
    workdir = os.getenv("TUNE_ORIG_WORKING_DIR")
    data = pickle.load(open(workdir + "/data.pckl", "rb"))
    model = RecurrentGCN(12, config["neurons"], config["filter_size"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    while True:
        cost = 0
        mse = 0
        datapoints = 0
        for _, snapshot in enumerate(data):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
            datapoints += 1
        cost = cost / (datapoints)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        for _, snapshot in enumerate(data):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            mse = mse + torch.mean((y_hat - snapshot.y) ** 2)
            datapoints += 1
        mse = mse / (datapoints)
        mse = mse.item()
        tune.report(mse=mse)


def main(num_samples=30, max_num_epochs=200, gpus_per_trial=0):
    config = {
        "neurons": tune.choice([8, 16, 32, 64, 128]),
        "lr": tune.loguniform(1e-4, 2e-1),
        "filter_size": tune.choice([1]),
    }

    scheduler = ASHAScheduler(
        metric="mse",
        mode="min",
        max_t=max_num_epochs,
        grace_period=4,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["mse", "training_iteration"])
    result = tune.run(
        train_full,
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("mse", "min", "last-5-avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final mse: {}".format(best_trial.last_result["mse"]))


if __name__ == "__main__":
    main(num_samples=100, max_num_epochs=100, gpus_per_trial=0)
