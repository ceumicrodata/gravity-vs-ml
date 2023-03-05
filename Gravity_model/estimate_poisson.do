use "temp/trade_analysis.dta", clear

local T1 2000
local T2 2019

generate prediction = .
forvalues t = `T1'/`T2' {
    poisson import ln_gdp_o ln_gdp_d ln_distw contig comlang_off comcol landlocked_o landlocked_d if year < `t'
    * this is a conditional prediction because it includes current GDP
    predict import_hat if year == `t', n
    replace prediction = int(import_hat) if year == `t'
    drop import_hat
}

drop if year < `T1'
keep iso_o iso_d year import prediction
export delimited "Gravity_model/prediction.csv", replace
