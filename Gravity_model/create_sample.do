use "temp/trade_edgelist.dta", clear

local nodevars gdp area total_population dis_int landlocked

rename iso_o cnum
merge m:1 cnum year using "temp/trade_nodelist.dta", keep(match) nogenerate
foreach X in `nodevars' {
    rename `X' `X'_o
}
rename cnum iso_o
rename iso_d cnum
merge m:1 cnum year using "temp/trade_nodelist.dta", keep(match) nogenerate
foreach X in `nodevars' {
    rename `X' `X'_d
}
rename cnum iso_d

save "temp/trade_sample.dta", replace
