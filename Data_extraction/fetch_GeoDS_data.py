# This code extracts and saves data from GeoDS

# encoding: utf-8
# copyright: GeoDS Lab, University of Wisconsin-Madison
# authors: Yuhao Kang, Song Gao, Jinmeng Rao

import requests
import os
import pandas as pd
from datetime import datetime

# Set parameters
start_year = '2019'
start_month = '01'
start_day = '07'
output_folder = '../Input_datasets/GeoDS_mobility_flow_prediction'
end_year = '2021'
end_month = '12'
end_day = '27'
ct = False
county = False
state = True

start_year = str(start_year).zfill(4)
start_month = str(start_month).zfill(2)
start_day = str(start_day).zfill(2)

if end_year == None:
    end_year = str(start_year).zfill(4)
else:
    end_year = str(end_year).zfill(4)
    
if end_month == None:
    end_month = str(start_month).zfill(2)
else:
    end_month = str(end_month).zfill(2)
    
if end_day == None:
    end_day = str(start_day).zfill(2)
else:
    end_day = str(end_day).zfill(2)

# Check if dates are valid
all_time = pd.date_range(start=f'2018-01-01', end=f'2025-01-06', freq="7D")
all_time = pd.DataFrame(all_time, columns=["date"])
all_time["date"] = all_time["date"].apply(lambda x: str(x).split(' ')[0])

if f'{start_year}-{start_month}-{start_day}'in all_time.values:
    is_valid_start = True
else:
    is_valid_start = False
    print("The start date is not a Monday. Please re-enter.")
if f'{end_year}-{end_month}-{end_day}'in all_time.values:
    is_valid_end = True
else:
    is_valid_end = False
    print("The end date is not a Monday. Please re-enter.")

# Download files of one day
def download_file(scale, year, month, day, output_folder):
    try:
        if os.path.exists(f"{output_folder}/") == False:
            os.mkdir(f"{output_folder}/")
        if os.path.exists(f"{output_folder}/{scale}/") == False:
            os.mkdir(f"{output_folder}/{scale}/")
        if scale == "ct2ct":
            if os.path.exists(f"{output_folder}/{scale}/{year}_{month}_{day}/") == False:
                os.mkdir(f"{output_folder}/{scale}/{year}_{month}_{day}/")
    except Exception as e:
        print(e)
        print("There is no output folder. Please create the output folder first!")               
                
    try:
        if scale == "ct2ct":
            for i in range(20):
                if year == "2019":
                    repo = "WeeklyFlows-Ct2019"
                elif year == "2020":
                    repo = "WeeklyFlows-Ct2020"
                elif year == "2021":
                    repo = "WeeklyFlows-Ct2021"
                r = requests.get(url=f"https://raw.githubusercontent.com/GeoDS/COVID19USFlows-{repo}/master/weekly_flows/{scale}/{year}_{month}_{day}/weekly_{scale}_{year}_{month}_{day}_{i}.csv")
                with open(f"{output_folder}/{scale}/{year}_{month}_{day}/weekly_{scale}_{year}_{month}_{day}_{i}.csv", 'wb') as file:
                    file.write(r.content)
        else:
            r = requests.get(url=f"https://raw.githubusercontent.com/GeoDS/COVID19USFlows-WeeklyFlows/master/weekly_flows/{scale}/weekly_{scale}_{year}_{month}_{day}.csv")
            with open(f"{output_folder}/{scale}/weekly_{scale}_{year}_{month}_{day}.csv", 'wb') as file:
                file.write(r.content)
        return True
    except Exception as e:
        print(e)
        return False

if (is_valid_start == True) and (is_valid_end == True):
    # Create time series dataframe
    time_df = pd.date_range(start=f'{start_year}-{start_month}-{start_day}', end=f'{end_year}-{end_month}-{end_day}', freq='7D')
    time_df = pd.DataFrame(time_df, columns=["date"])
    time_df["year"] = time_df["date"].apply(lambda x: str(x.year).zfill(4))
    time_df["month"] = time_df["date"].apply(lambda x: str(x.month).zfill(2))
    time_df["day"] = time_df["date"].apply(lambda x: str(x.day).zfill(2))

    # Download files at each scale
    if ct == True:
        time_df.apply(lambda x: download_file('ct2ct', x.year, x.month, x.day, output_folder), axis=1)
    if county == True:
        time_df.apply(lambda x: download_file('county2county', x.year, x.month, x.day, output_folder), axis=1)
    if state == True:
        time_df.apply(lambda x: download_file('state2state', x.year, x.month, x.day, output_folder), axis=1)

# Assign input folder path
input_folder = '../Input_datasets/GeoDS_mobility_flow_prediction/state2state/'
output_name = 'state2state_merged.csv'

if os.path.exists(f"{output_folder}") == False:
    os.mkdir(f"{output_folder}")

# Merge all files
flow_all = []
for file in os.listdir(input_folder):
    if file[-3:] == "csv":
        flow_df = pd.read_csv(f'{input_folder}/{file}')
        flow_df["start_date"] = flow_df["date_range"].apply(lambda x: datetime.strptime(x.split(' - ')[0], '%m/%d/%y'))
        flow_df = flow_df[["geoid_o", "geoid_d", "start_date", "visitor_flows", "pop_flows"]]

        flow_all.append(flow_df)


result = pd.concat([x for x in flow_all])

# Add zero flows
geo_ids = set(result["geoid_o"]) | set(result["geoid_d"])
start_dates = set(result["start_date"])
available_data = set(zip(result.geoid_o, result.geoid_d, result.start_date))
all_data = set([(origin, destination, time_period) for origin in geo_ids for destination in geo_ids for time_period in start_dates])
zero_flow_data = list(all_data - available_data)

zero_flow_df = pd.DataFrame(zero_flow_data, columns =['geoid_o', 'geoid_d', 'start_date'])
zero_flow_df['visitor_flows'] = 0
zero_flow_df['pop_flows'] = 0

result = pd.concat([result, zero_flow_df])
result = result.sort_values(by=['geoid_o', 'geoid_d', 'start_date'])

result['visitor_flows'] = result['visitor_flows'].astype('int64')
result['pop_flows'] = result['pop_flows'].astype('int64')

result.to_csv(output_folder + '/' + output_name, index=False)