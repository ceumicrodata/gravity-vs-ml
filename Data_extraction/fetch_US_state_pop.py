import pandas as pd
import requests
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
API_key  = os.environ['CENSUS_API_KEY']

# access the Population Estimates API and extract variables of interest
# provides overall population estimates and population densities
pop_url = f'https://api.census.gov/data/2019/pep/population?get=NAME,POP,DENSITY&for=state:*&key={API_key}'
response = requests.get(pop_url)
pop_data = response.json()
pop_df = pd.DataFrame(pop_data[1:], columns=pop_data[0]).rename(columns={'NAME':'state', 'state':'GEOID'})

pop_df.to_csv('../Input_datasets/Mobility_flow_prediction_shared/US_state_pop_2019_census.csv')