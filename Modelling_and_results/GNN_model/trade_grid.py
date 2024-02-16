from DCRNN_OLS import DataTransformerParameters, NeuralNetworkParameters, DCRNNEdgeEstimator
import pandas as pd
import numpy as np
import torch
from torch import nn

hidden_sizes = [3, 4, 5, 6, 7, 8]
losses = [nn.MSELoss, nn.L1Loss]
lrs = [0.03, 0.05, 0.01, 0.1, 0.005]


edges = pd.read_csv('../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv')
nodes = pd.read_csv('../Output_datasets/Yearly_trade_data_prediction/trade_nodelist.csv')
nodes.gdp = nodes.groupby('country').gdp.bfill()
predyears = range(2000, 2020)
for learning_rate in lrs:
    for lossfn in losses:
        for hidden_size in hidden_sizes:
            for predyear in predyears:
                dt_params = DataTransformerParameters('minmax')
                nn_params = NeuralNetworkParameters(lossfn, torch.optim.Adam, learning_rate, hidden_size, 1000, 42)
                model = DCRNNEdgeEstimator(nn_params,dt_params)
                model.fit((edges, nodes, predyear-5, predyear-1), 'trade')
                predictions = model.trade_post_predict_transform(model.predict((edges, nodes, [predyear]),'trade'), (edges, nodes, [predyear]))
                predictions.to_csv(f"predictions{learning_rate}_{hidden_size}_{int(lossfn==losses[0])}_{predyear}.csv", index=False)