use "temp/trade_edgelist.dta", clear

local nodevars gdp area total_population dis_int landlocked

rename iso_o iso_numeric
merge m:1 iso_numeric year using "temp/trade_nodelist.dta", keep(match) nogenerate
foreach X in `nodevars' {
    rename `X' `X'_o
}
rename iso_numeric iso_o
rename iso_d iso_numeric
merge m:1 iso_numeric year using "temp/trade_nodelist.dta", keep(match) nogenerate
foreach X in `nodevars' {
    rename `X' `X'_d
}
rename iso_numeric iso_d

* data does not include domestic trade
drop if iso_o == iso_d

save "temp/trade_sample.dta", replace
