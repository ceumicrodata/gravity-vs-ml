# This code extracts input data from Google mobility dataset downloaded from https://www.google.com/covid19/mobility/

import pandas as pd

years = [2020, 2021, 2022]
mobility_data_dict = {}
mobility_data_list = []

for year in years:
    mobility = pd.read_csv(f'../Input_datasets/Google_mobility_flow_prediction/Large_files/{year}_US_Region_Mobility_Report.csv', dtype=object)

    # Get state level info
    mobility = mobility[mobility["iso_3166_2_code"].isnull()==False]

    # US-DC (District of Columbia) is not state so remove
    #print(len(mobility_2020["iso_3166_2_code"].unique()))
    mobility = mobility[mobility["iso_3166_2_code"]!="US-DC"]

    # Drop columns that are not informative
    mobility.drop(columns=["census_fips_code", "sub_region_2", "metro_area", "place_id"], inplace=True)

    mobility_data_dict[year] = mobility
    mobility_data_list.append(mobility)

mobility_data = pd.concat(mobility_data_list)
mobility_data.sort_values(by=["sub_region_1", "date"], inplace=True)

mobility_data.to_csv("../Input_datasets/Google_mobility_flow_prediction/Google_mobility_data.csv", index=False)
