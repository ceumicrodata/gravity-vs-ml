#################################
# Class for compiling flow datasets
##################################

import torch
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class FlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 domain: str,
                 columns: dict,
                 nodes_edges: Optional[pd.DataFrame] = None,
                ) -> None:
        self.nodes_edges = nodes_edges
        self.domain = domain
        self.columns = columns
        self.node_id = columns['node_id']
        self.node_timestamp = columns['node_timestamp']
        self.node_features = columns['node_features'],
        self.flow_origin = columns['flow_origin']
        self.flow_destination = columns['flow_destination']
        self.flows_timestamp = columns['flows_timestamp']
        self.target_value = columns['target_value']
        self.flows_features = columns['flows_features']
                
        if self.target_value + '_target' in self.nodes_edges.columns:
            target = self.nodes_edges[[self.target_value + '_target']]
            features = self.nodes_edges.drop(columns=[self.target_value + '_target',
                                                        self.flow_destination,
                                                        self.flow_origin], axis=1)
            target = target.T.to_dict('list')
            self.target_dict = {}
            for key, value in target.items():
                self.target_dict[key] = torch.from_numpy(np.array(value)).float()

            features = features.T.to_dict('list')
            self.features_dict = {}
            for key, value in features.items():
                self.features_dict[key] = torch.from_numpy(np.array(value)).float()

    def add_past_values(self,
                        edge_columns: list,
                        node_columns: list,
                        periods: int =1) -> None:
        
        past_value_columns = []
        
        if self.domain!='Google':
            past_value_columns.extend([i+'_o' for i in node_columns] + [i+'_d' for i in node_columns])
        else:
            past_value_columns.extend(node_columns)
        past_value_columns.extend(edge_columns)

        for i in range(1, periods+1):
            for past_value_column in past_value_columns:
                self.nodes_edges[f'{past_value_column}_{i}'] = self.nodes_edges.groupby(
                    [self.flow_origin, self.flow_destination]
                )[past_value_column].shift(i)

        self.nodes_edges.dropna(inplace=True)

    def add_target_values(self) -> None:

        self.nodes_edges[f'{self.target_value}_target'] = self.nodes_edges.groupby(
            [self.flow_origin, self.flow_destination]
        )[self.target_value].shift(-1)
        self.nodes_edges[f'{self.target_value}_target'].fillna(0, inplace=True)

    def split_train_validate_test_scale(self,
                                  validation_period = 0.2,
                                  columns_to_scale = [],
                                  split_by_period_only = False,
                                  custom_seed = 1,
                                  scaling = 'MinMax') -> Tuple['FlowDataset', 'FlowDataset', 'FlowDataset']:

        columns_to_scale = [i for i in self.nodes_edges.columns if (
            i.startswith(tuple(columns_to_scale))
        ) & ('target' not in i)]
        
        test_period = self.nodes_edges[self.flows_timestamp].max()

        train_validation_data = self.nodes_edges[self.nodes_edges[
            self.flows_timestamp
        ]!=test_period].reset_index(drop=True)
        test_data = self.nodes_edges[self.nodes_edges[self.flows_timestamp]==test_period].reset_index(drop=True)

        random.seed(custom_seed)
        if split_by_period_only == False:
            random_validation_sample = random.sample(list(range(train_validation_data.shape[0])),
                                                     math.floor(train_validation_data.shape[0]*validation_period))
            train_data = train_validation_data[~train_validation_data.index.isin(
                random_validation_sample
            )].reset_index(drop=True)
            validation_data = train_validation_data[train_validation_data.index.isin(
                random_validation_sample
            )].reset_index(drop=True)
        else:
            timestamp_list = list(sorted(list(train_validation_data[self.flows_timestamp].unique())))
            validation_timestamps = timestamp_list[math.floor(len(timestamp_list)*validation_period):]
            train_data = train_validation_data[~train_validation_data[self.flows_timestamp].isin(
                random_validation_sample
            )].reset_index(drop=True)
            validation_data = train_validation_data[train_validation_data[self.flows_timestamp].isin(
                random_validation_sample
            )].reset_index(drop=True)
        
        if scaling == 'MinMax':
            scaler = MinMaxScaler()
            # Train data
            train_data[columns_to_scale] = train_data[columns_to_scale].astype(float)
            train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
            # Validation data
            validation_data[columns_to_scale] = validation_data[columns_to_scale].astype(float)
            validation_data[columns_to_scale] = scaler.transform(validation_data[columns_to_scale])
            # Test data
            test_data[columns_to_scale] = test_data[columns_to_scale].astype(float)
            test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])
        else: # Use logs
            # Train data
            train_data[columns_to_scale] = np.log1p(train_data[columns_to_scale])
            # Validation data
            validation_data[columns_to_scale] = np.log1p(validation_data[columns_to_scale])
            # Test data
            test_data[columns_to_scale] = np.log1p(test_data[columns_to_scale])

        return FlowDataset(domain=self.domain, columns=self.columns, nodes_edges = train_data), \
            FlowDataset(domain=self.domain, columns=self.columns, nodes_edges = validation_data), \
                FlowDataset(domain=self.domain, columns=self.columns, nodes_edges = test_data)

    def get_feature_dim(self) -> int:
        return self.nodes_edges.shape[1] - 3

    def compile_predictions(self, columns_to_rename: Dict[str, str] = {}) -> pd.DataFrame:

        self.nodes_edges['Timestamp_target'] = self.nodes_edges[self.flows_timestamp] +1
        self.nodes_edges.rename(columns=dict(
            {self.target_value +'_target':'target'}, **columns_to_rename
        ), inplace=True)
        return self.nodes_edges[['year','target', self.flow_origin, self.flow_destination,'prediction']]

    def __len__(self) -> int:
        return self.nodes_edges.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        return self.features_dict[index], self.target_dict[index]

