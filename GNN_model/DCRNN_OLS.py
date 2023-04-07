import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU
from torch_geometric_temporal.signal import (
    StaticGraphTemporalSignal,
)

from tqdm import tqdm

import pickle

from dataclasses import dataclass
from typing import Callable, Iterable, Any
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class DCRNNEdgePredictor(torch.nn.Module):
    """DCRNN-OLS joint predictor architecture"""

    def __init__(self, in_channels: int, out_channels: int, filter_size: int):
        """
        Initialize a pytorch model with DCRNN-OLS architecture

        Args:
            in_channels:
                Number of DCRNN input channels.
            out_channels:
                Number of DCRNN output channels - OLS hidden features
            filter_size:
                DCRNN filter size.
        """
        super().__init__()
        self.recurrent = DCRNN(in_channels, out_channels, filter_size)
        self.relu = nn.ReLU()
        self.linear = torch.nn.Linear(out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a forward pass

        Args:
            x:
                feature Pytorch Float Tensor
            edge_index:
                Pytorch Float Tensor of edge indices
            edge_weight:
                Pytorch Float Tensor of edge weights

        Returns:
            out:
                Pytorch Float Tensor for all edges
        """
        out = self.recurrent(x, edge_index, edge_weight)
        out = F.relu(out)
        stacked = torch.cat(
            [
                torch.cat(
                    [out, torch.cat([out[[i]] for _ in range(out.size()[0])], 0)], 1
                )
                for i in range(out.size()[0])
            ],
            0,
        )
        out = self.linear(stacked)
        return out


@dataclass
class NeuralNetworkParameters:
    """
    Dataclass containing neural network parameters

    Args:
        criterion:
            Callable returning an optimization criterion.
        optimizer:
            Callable accepting model parameters and a learning rate and returning an optimizer
        learning_rate:
            Learning rate for the optimizer
        hidden_size:
            Dimension of the hidden layer
        num_epochs:
            Number of training epochs
        random_seed:
            Random seed for reproducibility
    """

    criterion: Callable[[], nn.modules.loss._Loss]
    optimizer: Callable[[Iterable, float], torch.optim.Optimizer]
    learning_rate: float
    hidden_size: int
    num_epochs: int
    random_seed: int


@dataclass
class DataTransformerParameters:
    """
    Dataclass containing data transformer configuration

    Args:
        scaler_type:
            One of (minmax, standard, maxabs, robust)
    """

    scaler_type: str


@dataclass
class IdTransformer:
    """"""

    raw_to_nx: dict
    nx_to_raw: dict


@dataclass
class DCRNNEdgeEstimatorState:
    """
    Contains state of a FeedForwardEstimator

    Args:
        dt_params:
            Configuration of the data transformer
        nn_params:
            Configuration of the feed forward model
        scaler_x:
            Scaler of the features
        scaler_y:
            Scaler of the targets
        model:
            DCRNNEdgeEstimator model instance
        training_loss:
            List of training losses across training epochs
        validation_loss:
            List of validation losses across training epochs
    """

    dt_params: None | DataTransformerParameters
    nn_params: None | NeuralNetworkParameters
    scaler_x: None | MinMaxScaler | StandardScaler | MaxAbsScaler | RobustScaler
    scaler_y: None | MinMaxScaler | StandardScaler | MaxAbsScaler | RobustScaler
    model: None | nn.Module
    training_loss: None | list[torch.Tensor]
    validation_loss: None | list[torch.Tensor]
    id_transformer: None | IdTransformer


class DCRNNEdgeEstimator:
    """
    DCRNN-based edge weight estimator

    Args:
        nn_params:
            Parameters for the neural network model.
        dt_params:
            Parameters for data transformation.
        state:
            An optional DCRNNEdgeEstimatorState (to instantiate a fitted model).
    """

    def __init__(
        self,
        nn_params: NeuralNetworkParameters,
        dt_params: DataTransformerParameters,
        state: DCRNNEdgeEstimatorState = DCRNNEdgeEstimatorState(
            None, None, None, None, None, None, None, None
        ),
    ):
        self.state = state
        if not self.state.dt_params:
            self.state.dt_params = dt_params
        if not self.state.nn_params:
            self.state.nn_params = nn_params

    @staticmethod
    def get_scaler(
        scaler: str,
    ) -> MinMaxScaler | StandardScaler | MaxAbsScaler | RobustScaler:
        """
        Factory method for scalers

        Args:
            scaler:
                One of (minmax, standard, maxabs, robust)

        Returns:
            A sklearn scaler instance
        """
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    def create_node_id_transformers(
        self, nodes_df: pd.DataFrame, col: str
    ) -> IdTransformer:
        """Creates a node id transformer instance based on the input data"""
        original_ids = nodes_df[col].unique().tolist()
        return IdTransformer(
            dict(zip(original_ids, range(len(original_ids)))),
            dict(zip(range(len(original_ids)), original_ids)),
        )

    def trade_data_transformer(
        self, edges: pd.DataFrame, nodes: pd.DataFrame, start_year: int, end_year: int
    ) -> StaticGraphTemporalSignal:
        """Method for transforming trade data to PTGT data iterator"""
        edge_subset = edges[edges.Period.isin(range(start_year, end_year + 1))].copy()
        edge_subset["weight"] = 1 / np.log(edge_subset.distcap)
        assert edge_subset.shape[0] / edge_subset.iso_d.nunique() ** 2 == len(
            range(start_year, end_year + 1)
        )
        self.state.id_transformer = self.create_node_id_transformers(
            nodes, "iso_numeric"
        )
        edge_subset_single_year = edge_subset[edge_subset.Period == start_year].copy()
        edge_subset_single_year["source"] = edge_subset_single_year.iso_o.map(
            lambda id: self.state.id_transformer.raw_to_nx[id]
        )
        edge_subset_single_year["target"] = edge_subset_single_year.iso_d.map(
            lambda id: self.state.id_transformer.raw_to_nx[id]
        )
        edges = np.array(
            [
                edge_subset_single_year.source.tolist(),
                edge_subset_single_year.target.tolist(),
            ]
        )
        weights = np.array(edge_subset_single_year.weight.tolist())

        nodes["gdp"] = np.log(nodes["gdp"])
        nodes["total_population"] = np.log(nodes["total_population"])
        nodes["urban_population(%_of_total)"] = (
            nodes["urban_population(%_of_total)"] / 100
        )
        nodes["area"] = np.log(nodes["area"])
        nodes = nodes.join(
            pd.get_dummies(
                pd.DataFrame(
                    {
                        "continent": pd.Categorical(
                            nodes.continent, nodes.continent.unique()
                        )
                    },
                    index=nodes.index,
                )
            )
        )
        self.feature_set = [
            "gdp",
            "total_population",
            "urban_population(%_of_total)",
            "area",
        ] + [feat for feat in nodes if "continent_" in feat]
        features = []
        for year in tqdm(range(start_year, end_year), "Compiling features"):
            yearly_features = []
            for node in self.state.id_transformer.nx_to_raw.keys():
                node_feat = nodes.loc[
                    (
                        (nodes.year == year)
                        & (
                            nodes.iso_numeric
                            == self.state.id_transformer.nx_to_raw[node]
                        )
                    ),
                    self.feature_set,
                ].values.tolist()
                yearly_features.append(node_feat)
            features.append(yearly_features)

        features = np.array(features)

        target_subset = edge_subset[
            edge_subset.Period.isin(range(start_year + 1, end_year + 1))
        ].copy()
        target_subset["Value"] = np.log(target_subset["Value"] + 1)
        # target_subset['Value'] = self.apply_target_scaling(target_subset['Value'])
        targets = []
        for year in tqdm(range(start_year + 1, end_year + 1), "Compiling targets"):
            yearly_targets = []
            for origin in self.state.id_transformer.nx_to_raw.keys():
                for destination in self.state.id_transformer.nx_to_raw.keys():
                    yearly_targets.append(
                        target_subset.loc[
                            (
                                (
                                    target_subset.iso_o
                                    == self.state.id_transformer.nx_to_raw[origin]
                                )
                                & (
                                    target_subset.iso_d
                                    == self.state.id_transformer.nx_to_raw[destination]
                                )
                                & (target_subset.Period == year)
                            ),
                            "Value",
                        ].values[0]
                    )
            targets.append(yearly_targets)

        targets = np.array(targets)
        self.edges = edges
        self.weights = weights

        return StaticGraphTemporalSignal(edges, weights, features, targets)

    def trade_data_prediction_transformer(
        self, edges: pd.DataFrame, nodes: pd.DataFrame, prediction_years: list[int]
    ) -> StaticGraphTemporalSignal:
        """Method for generating PTGT data iterators for trade prediction"""
        edge_subset = edges[edges.Period.isin(prediction_years)].copy()
        edge_subset["weight"] = 1 / np.log(edge_subset.distcap)
        assert edge_subset.shape[0] / edge_subset.iso_d.nunique() ** 2 == len(
            prediction_years
        )
        edge_subset["source"] = edge_subset.iso_o.map(
            lambda id: self.state.id_transformer.raw_to_nx[id]
        )
        edge_subset["target"] = edge_subset.iso_d.map(
            lambda id: self.state.id_transformer.raw_to_nx[id]
        )

        nodes["gdp"] = np.log(nodes["gdp"])
        nodes["total_population"] = np.log(nodes["total_population"])
        nodes["urban_population(%_of_total)"] = (
            nodes["urban_population(%_of_total)"] / 100
        )
        nodes["area"] = np.log(nodes["area"])
        nodes = nodes.join(
            pd.get_dummies(
                pd.DataFrame(
                    {
                        "continent": pd.Categorical(
                            nodes.continent, nodes.continent.unique()
                        )
                    },
                    index=nodes.index,
                )
            )
        )
        self.feature_set = [
            "gdp",
            "total_population",
            "urban_population(%_of_total)",
            "area",
        ] + [feat for feat in nodes if "continent_" in feat]
        features = []
        for year in tqdm(prediction_years, "Compiling features"):
            yearly_features = []
            for node in self.state.id_transformer.nx_to_raw.keys():
                node_feat = nodes.loc[
                    (
                        (nodes.year == year - 1)
                        & (
                            nodes.iso_numeric
                            == self.state.id_transformer.nx_to_raw[node]
                        )
                    ),
                    self.feature_set,
                ].values.tolist()
                yearly_features.append(node_feat)
            features.append(yearly_features)

        features = np.array(features)

        target_subset = edge_subset
        target_subset["Value"] = np.log(target_subset["Value"] + 1)
        # target_subset['Value'] = pd.Series(self.state.scaler_y.transform(target_subset['Value'].values.reshape([-1,1]))[:,0], index=target_subset.index)
        targets = []
        for year in tqdm(prediction_years, "Compiling targets"):
            yearly_targets = []
            for destination in self.state.id_transformer.nx_to_raw.keys():
                for origin in self.state.id_transformer.nx_to_raw.keys():
                    yearly_targets.append(
                        target_subset.loc[
                            (
                                (
                                    target_subset.iso_o
                                    == self.state.id_transformer.nx_to_raw[origin]
                                )
                                & (
                                    target_subset.iso_d
                                    == self.state.id_transformer.nx_to_raw[destination]
                                )
                                & (target_subset.Period == year)
                            ),
                            "Value",
                        ].values[0]
                    )
            targets.append(yearly_targets)

        targets = np.array(targets)

        edges = self.edges
        weights = self.weights

        return StaticGraphTemporalSignal(edges, weights, features, targets)

    def apply_target_scaling(self, target: pd.Series) -> pd.Series:
        """Applies target scaling with fitting the scaler"""
        self.state.scaler_y = self.get_scaler(self.state.dt_params.scaler_type)
        scaled = self.state.scaler_y.fit_transform(target.values.reshape(-1, 1))
        return pd.Series(scaled[:, 0], index=target.index)

    def fit(self, raw_data: tuple[Any], data_type: str):
        """
        Fit an Estimator

        Args:
            raw_data:
                A tuple containing raw data sets.
            data_type:
                One of (trade, ). raw_data must correspond to this type.

        """
        if data_type == "trade":
            transformed_data = self.trade_data_transformer(*raw_data)

        torch.manual_seed(self.state.nn_params.random_seed)
        self.state.model = DCRNNEdgePredictor(
            len(self.feature_set), self.state.nn_params.hidden_size, 1
        )

        optimizer = self.state.nn_params.optimizer(
            self.state.model.parameters(), lr=self.state.nn_params.learning_rate
        )
        criterion = self.state.nn_params.criterion()

        self.state.model.train()
        with tqdm(total=self.state.nn_params.num_epochs) as progress_bar:
            for _ in range(self.state.nn_params.num_epochs):
                loss = 0
                periods = 0
                for _, snapshot in enumerate(transformed_data):
                    y_hat = self.state.model(
                        torch.flatten(snapshot.x, start_dim=1),
                        snapshot.edge_index,
                        snapshot.edge_attr,
                    )
                    loss = loss + criterion(torch.flatten(y_hat), snapshot.y)
                    periods += 1
                loss = loss / (periods)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if not self.state.training_loss:
                    self.state.training_loss = []
                self.state.training_loss.append(loss.item())
                progress_bar.set_description(
                    f"Optimization with backpropagation. Current loss: {loss.item():.4f}"
                )
                progress_bar.update(1)

    def validation_plot(self) -> matplotlib.figure.Figure:
        """
        Creates a plot from the training and validation losses against training epochs

        Returns:
            A matplotlib figure of the training and validation losses

        Raises:
            KeyError:
                If model has not been trained
        """
        raise NotImplementedError

    def training_loss_plot(self) -> matplotlib.figure.Figure:
        """
        Creates a plot from the training losses against training epochs

        Returns:
            A matplotlib figure of the training and validation losses

        Raises:
            KeyError:
                If model has not been trained
        """

        if not self.state.training_loss:
            raise KeyError("Fit first")
        plt.figure()
        plt.plot(self.state.training_loss)

        return plt.gcf()

    def predict(self, raw_data: pd.DataFrame, data_type: str) -> pd.Series:
        """
        Makes predictions

        Args:
            raw_data:
                A tuple containing raw data sets.
            data_type:
                One of (trade, ). raw_data must correspond to this type.

        Returns:
            predicted series

        Raises:
            ValueError:
                If model is not fitted
        """

        if data_type == "trade":
            transformed_data = self.trade_data_prediction_transformer(*raw_data)

        self.state.model.eval()
        predictions = []
        for _, snapshot in enumerate(transformed_data):
            y_hat = self.state.model(
                torch.flatten(snapshot.x, start_dim=1),
                snapshot.edge_index,
                snapshot.edge_attr,
            )
            predictions.append(y_hat)
        return predictions

    def trade_post_predict_transform(
        self, predictions: list[torch.Tensor], raw_data: tuple[Any]
    ) -> pd.DataFrame:
        """Inverse transformation and labelling of trade predictions"""
        edges, nodes, prediction_years = raw_data
        all_predictions = []
        for year, prediction in zip(prediction_years, predictions):
            preds = np.exp(prediction.detach().numpy()[:, 0])
            pred_df = pd.DataFrame(
                {
                    "prediction": preds,
                    "destination": [
                        self.state.id_transformer.nx_to_raw[i]
                        for i in self.state.id_transformer.nx_to_raw.keys()
                        for _ in range(len(self.state.id_transformer.nx_to_raw.keys()))
                    ],
                    "origin": [
                        self.state.id_transformer.nx_to_raw[i]
                        for _ in range(len(self.state.id_transformer.nx_to_raw.keys()))
                        for i in self.state.id_transformer.nx_to_raw.keys()
                    ],
                }
            )
            pred_df["year"] = year
            pred_df["target"] = (
                edges[edges.Period == year]
                .set_index(["iso_o", "iso_d"])
                .Value.loc[
                    zip(
                        pred_df.origin.values.tolist(),
                        pred_df.destination.values.tolist(),
                    )
                ]
                .values
            )
            all_predictions.append(pred_df)
        # return self.state.scaler_y.inverse_transform(preds.reshape(-1,1))
        return pd.concat(all_predictions)

    def to_pickle(self, path: str):
        """
        Saves the estimator state to pickle.

        Args:
            path:
                File path to be used
        """
        with open(path, "wb") as stream:
            pickle.dump(self.state, stream)

    def load_state_from_pickle(self, path: str):
        """
        Loads the estimator state from pickle.

        Args:
            path:
                File path to be used
        """
        with open(path, "rb") as stream:
            self.state = pickle.load(stream)
