# Importing the required libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd

# Downloading contents of the web page
url = "https://www.census.gov/geographies/reference-files/2010/geo/state-area.html"
data = requests.get(url).text

# Creating BeautifulSoup object
soup = BeautifulSoup(data, 'html.parser')

# Creating list with all tables
table = soup.find('table')

# Defining of the dataframe
df = pd.DataFrame(columns=['sub_region_1', 'Total_Area'])

# Collecting Ddata
for row in table.tbody.find_all('tr'):    
    # Find all data for each column
    columns = row.find_all('td')
    
    if(columns != []):
        state = columns[0].text.strip()
        total_area = columns[2].text.strip()

        df = df.append({'sub_region_1': state, 'Total_Area': total_area}, ignore_index=True)

df = df.drop([0,1,2])

# State name mapping table
state_mapping_table = pd.read_csv('../Input_datasets/Mobility_flow_prediction_shared/us_states_w_fips_and_coordinates.csv')

df = pd.merge(df, state_mapping_table[['sub_region_1', 'FIPS']], on='sub_region_1')

df.to_csv("../Input_datasets/Mobility_flow_prediction_shared/US_state_sizes.csv")