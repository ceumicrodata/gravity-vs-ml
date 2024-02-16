import pandas as pd
import io
import requests
from retrying import retry
import pickle
from collections import defaultdict
import area
import tqdm

################################
# Useful functions
#################################

def retry_if_connection_error(exception):
    return isinstance(exception, ConnectionError)

def retry_if_status_code_not_200(result):
    return result.status_code!=200

# if exception retry with 2 second wait  
@retry(retry_on_exception=retry_if_connection_error, retry_on_result=retry_if_status_code_not_200, wait_fixed=2000)
def safe_request(url, **kwargs):
    return requests.get(url, **kwargs)


######################
# Obtain US state ids
######################
overpass_url = "https://overpass-api.de/api/interpreter"

# Residential lines
overpass_query = f"""
[out:csv(::id,"ISO3166-2",name)];
rel[boundary=administrative][admin_level=4]["ISO3166-2"~"^US"];
map_to_area;
out tags;
"""
response = requests.get(overpass_url, params={'data': overpass_query})

us_state_names = pd.read_csv(io.StringIO(response.text))['@id\tISO3166-2\tname'].str.split('\t',expand=True)
us_state_names.columns = ['overpass_id', 'state_short', 'state']
us_state_names = us_state_names.sort_values('state')

# Not states:
# American Samoa, District of Columbia,
# Guam, Northern Mariana Islands,
# Puerto Rico, United States Virgin Islands
us_state_names = us_state_names[(us_state_names['state']!='American Samoa') &
                                (us_state_names['state']!='District of Columbia') &
                                (us_state_names['state']!='Guam') &
                                (us_state_names['state']!='Northern Mariana Islands') &
                                (us_state_names['state']!='Puerto Rico') &
                                (us_state_names['state']!='United States Virgin Islands')]

state_overpass_ids = list(us_state_names['overpass_id'])

#print(len(state_overpass_ids))

feature_query_dict = {}

####################
# roads (3 features): total length (in km)
####################
# residential
types = ['residential']
query_string = ""
for type in types:
    query_string+=f"way(area.searchArea)[highway={type}]; "
feature_query_dict['residential_roads'] = query_string

# other
types = ['primary', 'secondary', 'tertiary', 'unclassified', 'service', 'primary_link',
'secondary_link', 'tertiary_link','living_street', 'pedestrian', 'track', 'road']
query_string = ""
for type in types:
    query_string+=f"way(area.searchArea)[highway={type}]; "
feature_query_dict['other_roads'] = query_string

# main
main_road_type = ['motorway', 'trunk', 'motorway_link', 'trunk_link']
query_string = ""
for type in types:
    query_string+=f"way(area.searchArea)[highway={type}]; "
feature_query_dict['main_roads'] = query_string

################
# land use (5 features): total area (in km2)
################
# residential
types = ['residential']
query_string = ""
for type in types:
    query_string+=f"relation(area.searchArea)[landuse={type}]; "
feature_query_dict['residential_land_use'] = query_string
# commercial
types = ['commercial']
query_string = ""
for type in types:
    query_string+=f"relation(area.searchArea)[landuse={type}]; "
feature_query_dict['commercial_land_use'] = query_string
# industrial
types = ['industrial', 'garages', 'port', 'quarry']
query_string = ""
for type in types:
    query_string+=f"relation(area.searchArea)[landuse={type}]; "
feature_query_dict['industrial_land_use'] = query_string
# retail
types = ['retail']
query_string = ""
for type in types:
    query_string+=f"relation(area.searchArea)[landuse={type}]; "
feature_query_dict['retail_land_use'] = query_string
# natural
landuse_types = ['farmland', 'farmyard', 'forest', 'grass',
        'greenfield', 'greenhouse_horticulture',
        'meadow', 'orchard', 'plant_nursery',
        'recreation_ground', 'village_green', 'vineyard']
leisure_types = ['park', 'garden', 'common', 'dog_park','nature_reserve', 'playground']
boundary_types = ['national_park', 'protected_area']
building_types = ['greenhouse']
query_string = ""
for type in landuse_types:
    query_string+=f"relation(area.searchArea)[landuse={type}]; "
for type in leisure_types:
    query_string+=f"relation(area.searchArea)[leisure={type}]; "
for type in boundary_types:
    query_string+=f"relation(area.searchArea)[boundary={type}]; "
for type in building_types:
    query_string+=f"relation(area.searchArea)[building={type}]; "
feature_query_dict['natural_land_use'] = query_string

################
# transport (2 features): count
################
# point
amenity_types = ['bus_station', 'car_rental', 'ferry_terminal']
public_transport_types = ['station', 'platform']
query_string = ""
for type in amenity_types:
    query_string+=f"node(area.searchArea)[amenity={type}]; "
for type in public_transport_types:
    query_string+=f"node(area.searchArea)[public_transport={type}]; "
feature_query_dict['point_transport'] = query_string
# building
amenity_types = ['bus_station', 'car_rental', 'ferry_terminal']
building_types = ['train_station', 'transportation', 'parking']
public_transport_types = ['station', 'platform']
query_string = ""
for type in amenity_types:
    query_string+=f"relation(area.searchArea)[amenity={type}]; "
for type in building_types:
    query_string+=f"relation(area.searchArea)[building={type}]; "
for type in public_transport_types:
    query_string+=f"relation(area.searchArea)[public_transport={type}]; "
feature_query_dict['building_transport'] = query_string

################
# food (2 features): count
################
# point
amenity_types = ['bar', 'biergarten', 'cafe', 'fast_food',
                                       'food_court', 'ice_cream', 'pub', 'restaurant']
shop_types = ['alcohol', 'bakery', 'beverages',
                                    'brewing_supplies', 'butcher', 'cheese',
                                    'chocolate', 'coffee', 'confectionery',
                                    'convenience', 'deli', 'dairy', 'farm',
                                    'frozen_food', 'greengrocer', 'health_food',
                                    'ice_cream', 'organic', 'pasta', 'pastry',
                                    'seafood', 'spices', 'tea', 'water',
                                    'department_store', 'general', 'kiosk', 'mall',
                                    'supermarket', 'wholesale']
query_string = ""
for type in amenity_types:
    query_string+=f"node(area.searchArea)[amenity={type}]; "
for type in shop_types:
    query_string+=f"node(area.searchArea)[shop={type}]; "
feature_query_dict['point_food'] = query_string
# building
amenity_types = ['bar', 'biergarten', 'cafe', 'fast_food',
                                      'food_court', 'ice_cream', 'pub', 'restaurant']
shop_types = ['alcohol', 'bakery', 'beverages',
                                   'brewing_supplies', 'butcher', 'cheese',
                                   'chocolate', 'coffee', 'confectionery',
                                   'convenience', 'deli', 'dairy', 'farm',
                                   'frozen_food', 'greengrocer', 'health_food',
                                   'ice_cream', 'organic', 'pasta', 'pastry',
                                   'seafood', 'spices', 'tea', 'water',
                                   'department_store', 'general', 'kiosk', 'mall',
                                   'supermarket', 'wholesale']
query_string = ""
for type in amenity_types:
    query_string+=f"relation(area.searchArea)[amenity={type}]; "
for type in shop_types:
    query_string+=f"relation(area.searchArea)[shop={type}]; "
feature_query_dict['building_food'] = query_string

################
# health (2 features): count
################
# point
types = ['clinic', 'dentist', 'doctors', 'hospital',
        'pharmacy', 'social_facility', 'veterinary']
query_string = ""
for type in types:
    query_string+=f"node(area.searchArea)[amenity={type}]; "
feature_query_dict['point_health'] = query_string
# building
amenity_types = ['clinic', 'dentist', 'doctors', 'hospital',
                                        'pharmacy', 'social_facility', 'veterinary']
building_types = ['hospital']
query_string = ""
for type in amenity_types:
    query_string+=f"relation(area.searchArea)[amenity={type}]; "
for type in building_types:
    query_string+=f"relation(area.searchArea)[building={type}]; "
feature_query_dict['building_health'] = query_string

################
# education (2 features): count
################
# point
types = ['college', 'kindergarten', 'library',
        'school', 'university', 'research_institute',
        'music_school', 'language_school']
query_string = ""
for type in types:
    query_string+=f"node(area.searchArea)[amenity={type}]; "
feature_query_dict['point_education'] = query_string
# building
amenity_types = ['college', 'kindergarten', 'library',
                'school', 'university', 'research_institute',
                'music_school', 'language_school']
building_types = ['kindergarten', 'school', 'university']
query_string = ""
for type in amenity_types:
    query_string+=f"relation(area.searchArea)[amenity={type}]; "
for type in building_types:
    query_string+=f"relation(area.searchArea)[building={type}]; "
feature_query_dict['building_education'] = query_string

################
# retail (2 features): count
################
# point
shop_types = ['alcohol', 'bakery', 'beverages',
                                      'brewing_supplies', 'butcher', 'cheese',
                                      'chocolate', 'coffee', 'confectionery',
                                      'convenience', 'deli', 'dairy', 'farm',
                                      'frozen_food', 'greengrocer', 'health_food',
                                      'ice_cream', 'organic', 'pasta', 'pastry',
                                      'seafood', 'spices', 'tea', 'water',
                                      'department_store', 'general', 'kiosk', 'mall',
                                      'supermarket', 'wholesale']
amenity_type = ['marketplace', 'post_office']
highway_types = ['rest_area']
query_string = ""
for type in shop_types:
    query_string+=f"node(area.searchArea)[shop={type}]; "
for type in amenity_type:
    query_string+=f"node(area.searchArea)[amenity={type}]; "
for type in highway_types:
    query_string+=f"node(area.searchArea)[highway={type}]; "
feature_query_dict['point_retail'] = query_string
# building
shop_types = ['alcohol', 'bakery', 'beverages',
                                     'brewing_supplies', 'butcher', 'cheese',
                                     'chocolate', 'coffee', 'confectionery',
                                     'convenience', 'deli', 'dairy', 'farm',
                                     'frozen_food', 'greengrocer', 'health_food',
                                     'ice_cream', 'organic', 'pasta', 'pastry',
                                     'seafood', 'spices', 'tea', 'water',
                                     'department_store', 'general', 'kiosk', 'mall',
                                     'supermarket', 'wholesale']
amenity_type = ['marketplace', 'post_office']
highway_types = ['rest_area']
query_string = ""
for type in shop_types:
    query_string+=f"relation(area.searchArea)[shop={type}]; "
for type in amenity_type:
    query_string+=f"relation(area.searchArea)[amenity={type}]; "
for type in highway_types:
    query_string+=f"relation(area.searchArea)[highway={type}]; "
feature_query_dict['building_retail'] = query_string

#print(len(feature_query_dict.keys()))
#print(feature_query_dict.keys())

response_data_dict = defaultdict(dict)
final_data_dict = defaultdict(dict)

for overpass_id in tqdm.tqdm(state_overpass_ids):
  
  # land use TBA (5 features)
  for land_use_type in ['residential_land_use', 'commercial_land_use', 'industrial_land_use', 'retail_land_use', 'natural_land_use']:
    overpass_query = f"""
    [out:json];
    area({overpass_id})->.searchArea;
    (
    {feature_query_dict[land_use_type] }
    );
    out geom;
    """
    response = safe_request(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    areas = 0
    for element in data['elements']:
        for member in element['members']:
            try:
                vertices = []
                for position in member['geometry']:
                    vertices.append([position['lat'], position['lon']])
                areas += area.area({ "type": "Polygon", "coordinates": [vertices]}) / 1e6
            except KeyError:
                pass
    
    response_data_dict[overpass_id][land_use_type] = data
    final_data_dict[overpass_id][land_use_type+'_area'] = areas

  # roads
  for road_type in ['residential_roads', 'other_roads', 'main_roads']:
    overpass_query = f"""
    [out:json];
    area({overpass_id})->.searchArea;
    (
      {feature_query_dict[road_type] }
    );
    make stat number=count(ways),length=sum(length());
    out;
    """

    response = safe_request(overpass_url, params={'data': overpass_query})
    data = response.json()

    response_data_dict[overpass_id][road_type] = data
    #final_data_dict[overpass_id][road_type+'number'] = data['elements'][0]['tags']['number']
    try:
      final_data_dict[overpass_id][road_type+'_length'] = data['elements'][0]['tags']['length'] 
    except:
      print(overpass_id, road_type)

  # all other
  all_other_list = ['point_transport', 'building_transport', 'point_food', 'building_food',
                    'point_health', 'building_health', 'point_education', 'building_education',
                    'point_retail', 'building_retail']
  for type in all_other_list:
    overpass_query = f"""
    [out:json];
    area({overpass_id})->.searchArea;
    (
      {feature_query_dict[type]}
    );
    out count;
    """
    response = safe_request(overpass_url, params={'data': overpass_query})
    data = response.json()

    response_data_dict[overpass_id][type] = data

    try:
      final_data_dict[overpass_id][type] = data['elements'][0]['tags']['total']
    except:
      print(overpass_id, type)

with open('../Input_datasets/Mobility_flow_prediction_shared/Large_files/response_data_dict.pickle', 'wb') as handle:
    pickle.dump(response_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../Input_datasets/Mobility_flow_prediction_shared/Large_files/final_data_dict.pickle', 'wb') as handle:
    pickle.dump(final_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('../Input_datasets/Mobility_flow_prediction_shared/Large_files/response_data_dict.pickle', 'rb') as f:
#    response_data_dict = pickle.load(f)

#with open('../Input_datasets/Mobility_flow_prediction_shared/Large_files/final_data_dict.pickle', 'rb') as f:
#    final_data_dict = pickle.load(f)

final_df = pd.DataFrame(final_data_dict).transpose().fillna(0).reset_index().rename(columns={'index':'overpass_id'})
final_df = pd.merge(final_df, us_state_names, on='overpass_id')
final_df.to_csv("../Input_datasets/Mobility_flow_prediction_shared/overpass_features.csv")