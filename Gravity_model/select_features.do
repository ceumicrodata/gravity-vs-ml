use "temp/trade_sample.dta", clear

local logarithm gdp_o gdp_d total_population_o total_population_d distw 
local dummies landlocked_o landlocked_d contig comlang_off comcol

foreach X in `logarithm' {
    generate ln_`X' = ln(`X')
}

keep iso_? year import ln_* `dummies'
compress

save "temp/trade_analysis.dta", replace