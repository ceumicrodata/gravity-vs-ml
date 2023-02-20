import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from haversine import haversine
from tqdm import tqdm
import time

def get_coordinates(state_name):
    try:
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(f"{state_name} state", timeout=10)
        return (location.latitude, location.longitude)
    except GeocoderTimedOut:
        print(f"Timed out {state_name}")
        time.sleep(5)
        return get_coordinates(state_name)
    
tqdm.pandas()


df = pd.read_csv("data/2022_US_Region_Mobility_Report.csv")
states = df[['sub_region_1','iso_3166_2_code']].dropna().drop_duplicates()
states['coordinate'] = states['iso_3166_2_code'].progress_map(lambda state: get_coordinates(state))
cross_joined = states[['iso_3166_2_code','coordinate']].merge(states[['iso_3166_2_code','coordinate']], how='cross')
cross_joined['distance'] = cross_joined.apply(lambda r: haversine(r.coordinate_x, r.coordinate_y), axis=1)
cross_joined[['iso_3166_2_code_x','iso_3166_2_code_y','distance']].set_index(['iso_3166_2_code_x','iso_3166_2_code_y']).unstack().to_csv("US_state_adjacency.csv")
