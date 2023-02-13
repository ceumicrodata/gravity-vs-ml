import pandas as pd
df = pd.read_stata('http://www.cepii.fr/distance/dist_cepii.dta')
df.to_csv('../Input_datasets/cepii_edge.csv', index=False)
df = pd.read_stata('http://www.cepii.fr/distance/geo_cepii.dta')
df.to_csv('../Input_datasets/cepii_node.csv', index=False)