#################################
# Class for compiling flow datasets
##################################

import torch
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

class FlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 columns: dict,
                 unit: list,
                 nodes: Optional[pd.DataFrame] = None,
                 edges: Optional[pd.DataFrame] = None,
                 data_dict: Optional[dict] = None
                ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.columns = columns
        self.node_id = columns['node_id']
        self.node_timestamp = columns['node_timestamp']
        self.flow_origin = columns['flow_origin']
        self.flow_destination = columns['flow_destination']
        self.flows_timestamp = columns['flows_timestamp']
        self.flows_value = columns['flows_value']

        self.unit = unit

        if not data_dict:
            self.data_dict = self._to_dict(unit=unit)
        else:
            self.data_dict = data_dict
        self._update_dict_attributes()

    def _update_dict_attributes(self) -> None:
        self.data_dict_index_mapper = dict(enumerate(self.data_dict.keys()))
        self.id_list = list(sorted(set([key[0] for key in self.data_dict.keys()])))
        self.period_list = list(sorted(set([key[1] for key in self.data_dict.keys()])))

    def _merge_nodes_edges(self, nodes:pd.DataFrame, edges:pd.DataFrame,
                           node_columns:list, flow_columns:list
                           ) -> pd.DataFrame:
        
        nodes.rename(columns={node_columns[0]: flow_columns[0], node_columns[1]: flow_columns[2]}, inplace=True)
        nodes[flow_columns[1]] = nodes[flow_columns[0]]
        
        return pd.merge(pd.merge(edges, nodes.drop(flow_columns[1], axis=1), how='left', on=[flow_columns[0], flow_columns[2]]),
                        nodes.drop(flow_columns[0], axis=1), how='left', on=[flow_columns[1], flow_columns[2]], 
                        suffixes=('_o', '_d'))

    def _to_dict(self, unit:list) -> dict[str, pd.DataFrame]:

        merged_nodes_edges = self._merge_nodes_edges(self.nodes, self.edges, [self.node_id, self.node_timestamp],
                                                     [self.flow_origin, self.flow_destination, self.flows_timestamp])
        merged_nodes_edges.set_index(unit, inplace=True)
        merged_nodes_edges.sort_index(inplace=True)
        data_dict = {}
        for index in merged_nodes_edges.index.unique():
            data_dict[index] = merged_nodes_edges.loc[index]

        return data_dict

    def create_chunks(self,
                      chunk_size: int = 6) -> list['FlowDataset']:

        chunk_period_list = [self.period_list[i:i+chunk_size] for i in range(0, len(self.period_list)-(chunk_size-1))]
        chunk_list = []

        for chunk_period in chunk_period_list:
            chunk_data = {}
            for period in chunk_period:
                chunk_data.update({(id, period):self.data_dict[(id, period)] for id in self.id_list})
            chunk_data = FlowDataset(columns=self.columns,
                        unit = [self.flow_origin, self.flows_timestamp],
                        data_dict = chunk_data)
            chunk_list.append(chunk_data)
        return chunk_list
    
    def add_past_values(self,
                        edge_columns: list,
                        node_columns: list,
                        periods: int =1) -> None:
        
        node_columns = [i+'_o' for i in node_columns] + [i+'_d' for i in node_columns] 
        data_dict = {}
        for id in self.id_list:
            for period in self.period_list[periods:]:
                data = self.data_dict[(id, period)].copy()
                for past_period in range(1,periods+1):
                    data = pd.merge(data,
                                    self.data_dict[(id, period-past_period)][edge_columns + node_columns
                                                                            + [self.flow_destination]],
                                    how = 'left',
                                    on = [self.flow_destination],
                                    suffixes=('', f'_{past_period}'))
                data_dict[(id, period)] = data
        self.data_dict = data_dict
        self._update_dict_attributes()

    def add_target_values(self) -> None:

        data_dict = {}
        for id in self.id_list:
            for period in self.period_list[:-1]:
                data = self.data_dict[(id, period)].copy()
                data = pd.merge(data,
                                self.data_dict[(id, period+1)][[self.flows_value, self.flow_destination]],
                                how = 'left',
                                on = [self.flow_destination],
                                suffixes=('', '_target'))
                data_dict[(id, period)] = data
        self.data_dict = data_dict
        self._update_dict_attributes()

    def split_train_test(self,
                         test_period = 1) -> Tuple['FlowDataset', 'FlowDataset']:
        
        train_data = {(id, period): self.data_dict[(id, period)] for id in self.id_list for period in self.period_list[:-test_period]}
        test_data = {(id, period): self.data_dict[(id, period)] for id in self.id_list for period in self.period_list[-test_period:]}

        return FlowDataset(columns=self.columns,
                           unit = [self.flow_origin, self.flows_timestamp],
                           data_dict = train_data), FlowDataset(columns=self.columns,
                                                                unit = [self.flow_origin, self.flows_timestamp],
                                                                data_dict = test_data)

    def get_feature_dim(self) -> int:
        return self.data_dict[self.data_dict_index_mapper[0]].shape[1] - 2
    
    #def get_features_labels(self) -> None:

    def compile_predictions(self, columns_to_rename: Dict[str, str] = {}) -> pd.DataFrame:
        prediction_list = []
        for key in self.data_dict.keys():
            predicted_data = self.data_dict[key][[self.flow_destination, self.flows_value +'_target', 'prediction']].copy()
            predicted_data['Timestamp_target'] = key[1] + 1
            predicted_data[self.flow_origin] = key[0]
            prediction_list.append(predicted_data)

        predicted_data = pd.concat(prediction_list, axis=0)
        predicted_data.rename(columns=dict({self.flows_value +'_target':'target'}, **columns_to_rename), inplace=True)
        return predicted_data[['year','target','iso_o','iso_d','prediction']]

    def __len__(self) -> int:
        return len(self.data_dict.keys())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target = self.data_dict[self.data_dict_index_mapper[index]][self.flows_value + '_target']
        features = self.data_dict[self.data_dict_index_mapper[index]].drop(columns=[self.flows_value + '_target',
                                                                                    self.flow_destination], axis=1)

        target = torch.from_numpy(np.swapaxes(np.array([target]),0,1)).float()
        features = torch.from_numpy(np.array(features)).float()
                                  
        return features, target

    

    
    