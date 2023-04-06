use "temp/trade_analysis.dta", clear

local T1 2000
local T2 2014
* time windows to be used for training and testing
local trn_beg 0 
local trn_end 3
local tst_beg 4 
local tst_end 4

egen i = group(iso_o iso_d)
xtset i year
generate Fimport = F.import
generate prediction = .
forvalues t = `T1'/`T2' {
    poisson Fimport ln_gdp_o ln_gdp_d ln_distw contig comlang_off comcol landlocked_o landlocked_d if inrange(year - `t', `trn_beg', `trn_end')
    * this is a conditional prediction because it includes current GDP
    predict import_hat if inrange(year - `t', `tst_beg', `tst_end'), n
    replace prediction = L.import_hat if inrange(year - `t' - 1, `tst_beg', `tst_end')
    drop import_hat
}

drop if year <= `T1' + `tst_beg'
keep iso_o iso_d year import prediction
sort iso_o iso_d year
rename import target
export delimited "Gravity_model/prediction.csv", replace
