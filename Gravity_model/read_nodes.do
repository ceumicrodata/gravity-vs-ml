import delimited "temp/trade_nodelist.csv", case(preserve) encoding("utf-8") clear

duplicates drop cnum year, force
keep cnum country year gdp area total_population dis_int landlocked

save "temp/trade_nodelist.dta", replace