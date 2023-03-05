import delimited "temp/trade_edgelist.csv", case(preserve) encoding("utf-8") clear

rename Period year
rename Value import
drop v1

save "temp/trade_edgelist.dta", replace