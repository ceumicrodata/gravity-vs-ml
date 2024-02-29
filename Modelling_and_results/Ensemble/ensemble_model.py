import pandas as pd

# Trade ensemble
nn = pd.read_csv("../Neural_network/trade_predictions/gcp_limited_specification_prediction_2023-09-14_11_58_46.csv")
rf = pd.read_csv("../Random_forest/Results/RF_trade_limited_prediction_2023-10-10_20-09.csv")
tl = pd.read_csv("../Gravity_model/trade_lag_prediction.csv")

nn.rename(columns={"prediction": "prediction_dg", "year":"Period"}, inplace = True)
rf.rename(columns={"prediction": "prediction_rf", "year":"Period"}, inplace = True)
tl.rename(columns={"prediction": "prediction_tl", "year":"Period"}, inplace = True)

full_df = pd.merge(pd.merge(nn, rf, on=["Period", "iso_o", "iso_d"]), tl, on=["Period", "iso_o", "iso_d"])

full_df["prediction"] =  (1 * full_df["prediction_dg"] + \
                           1 * full_df["prediction_rf"] + \
                           1 * full_df["prediction_tl"])/3

full_df.rename(columns={"Period":"year"}, inplace=True)

full_df.to_csv("trade_ensemble_prediction.csv")

# GeoDS ensemble
nn = pd.read_csv("../Neural_network/GeoDS_mobility_predictions/GCP_prediction_limited_lag_2023-09-24_18_23_15.csv")
rf = pd.read_csv("../Random_forest/Results/RF_GeoDS_limited_lags_prediction_2023-10-10_19-33.csv")
tl = pd.read_csv("../Gravity_model/geods_lag_prediction.csv")

nn.rename(columns={"prediction": "prediction_dg", "year":"Period"}, inplace = True)
rf.rename(columns={"prediction": "prediction_rf", "year":"Period"}, inplace = True)
tl.rename(columns={"prediction": "prediction_tl", "year":"Period"}, inplace = True)

full_df = pd.merge(pd.merge(nn, rf, on=["Period", "origin", "destination"]), tl, on=["Period", "origin", "destination"])

full_df["prediction"] =  (1 * full_df["prediction_dg"] + \
                           1 * full_df["prediction_rf"] + \
                           1 * full_df["prediction_tl"])/3

full_df.rename(columns={"Period":"year"}, inplace=True)

full_df.to_csv("GeoDS_ensemble_prediction.csv")

# Google ensemble
nn = pd.read_csv("../Neural_network/Google_mobility_predictions/gcp_Google_prediction_limited_lags.csv")
rf = pd.read_csv("../Random_forest/Results/RF_Google_full_prediction_2023-10-10_19-48.csv")

nn.rename(columns={"prediction": "prediction_dg", "year":"Period"}, inplace = True)
rf.rename(columns={"prediction": "prediction_rf", "year":"Period"}, inplace = True)

full_df = pd.merge(nn, rf, on=["Period", "origin", "destination"])

full_df["prediction"] =  (1 * full_df["prediction_dg"] + \
                           1 * full_df["prediction_rf"])/2

full_df.rename(columns={"Period":"year"}, inplace=True)

full_df.to_csv("Google_ensemble_prediction.csv")