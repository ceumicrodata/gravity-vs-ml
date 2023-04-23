args inf outf

do "Gravity_model/read.do" "`inf'"
do "Gravity_model/select_features.do"

egen i = group(iso_o iso_d)
xtset i year
generate Fimport = F.import
* poisson often does not converge, use ppmlhdfe instead
ppmlhdfe Fimport ln_gdp_o ln_gdp_d ln_dist contig comlang_off comcol

summarize year
keep if year == r(max)
replace year = year+1
predict prediction, mu
replace prediction = int(prediction)

keep iso_o iso_d year import prediction
sort iso_o iso_d year
rename import target
export delimited "`outf'", replace
