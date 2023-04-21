args inf outf
use "`inf'"

do "Gravity_model/select_features.do"

egen i = group(iso_o iso_d)
xtset i year
generate Fimport = F.import
generate prediction = .
poisson Fimport ln_gdp_o ln_gdp_d ln_dist contig comlang_off comcol landlocked_o landlocked_d

predict import_hat if inrange(year - `t', `tst_beg', `tst_end'), n
replace prediction = L.import_hat if inrange(year - `t' - 1, `tst_beg', `tst_end')
drop import_hat

drop if year <= `T1' + `tst_beg'
keep iso_o iso_d year import prediction
sort iso_o iso_d year
rename import target
export delimited "`outf'", replace
