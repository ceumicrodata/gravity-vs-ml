### This code downloads macroeconomic variables from Worldbank data.

import wbgapi as wb
import pandas as pd
import numpy as np

### We restrict the dataset to the countries to be used in the analysis
nodes = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/cepii_node.csv')

countries_restricted = pd.read_csv('../Input_datasets/Yearly_trade_data_prediction/country_names_with_annual_trade_data.csv',usecols =['ISO_3166-1_numeric_code', 'Country_name'])

countries_restricted_merged = pd.merge(countries_restricted,nodes, how='left', left_on='ISO_3166-1_numeric_code', right_on='cnum')

countries = np.unique(nodes['iso3'].tolist()) 

### Add Romania separately as country code differs in Worldbank data
missing_countries = ['ROU']
if missing_countries not in countries:
    countries = np.append(countries,missing_countries)

restricted_raw = wb.data.DataFrame(['NY.GDP.MKTP.KD', 'SP.POP.TOTL','SP.URB.TOTL.IN.ZS'], countries, range (1995,2020))

derived_df = restricted_raw.stack().unstack('series')
derived_df.columns = ["gdp", "total_population","urban_population(%_of_total)"]

derived_df.to_csv('../Input_datasets/Yearly_trade_data_prediction/WBG_data_all_countries.csv')