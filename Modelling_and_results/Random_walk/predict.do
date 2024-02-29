import delimited using "Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv", case(preserve) clear
rename Value target
rename Period year
egen group = group(iso_o iso_d)
xtset group year
generate prediction = L.target
keep if year >= 2000
keep iso_o iso_d year prediction
export delimited using "Random_walk/trade_prediction.csv", replace

import delimited using "Output_datasets/GeoDS_mobility_flow_prediction/edge_target_list.csv", case(preserve) clear
rename pop_flows target
rename Timeline year
egen group = group(origin destination)
xtset group year
generate prediction = L.target
keep if year >= 50
keep if year/10 == int(year/10)
keep origin destination year prediction
export delimited using "Random_walk/GeoDS_prediction.csv", replace

import delimited using "Output_datasets/Google_mobility_flow_prediction/node_target_list.csv", case(preserve) clear
rename Value target
rename Timeline year
egen group = group(origin destination)
xtset group year
generate prediction = L.target
keep if inrange(year, 350, 950)
keep if year/50 == int(year/50)
keep origin destination year prediction
export delimited using "Random_walk/Google_prediction.csv", replace
