args inf outf

do "Gravity_model/read.do" "`inf'"
do "Gravity_model/select_features.do" "`inf'"

* select regression model based on filename
local model = substr("`outf'", strpos("`outf'", "/")+1, .)
local model = substr("`model'", 1, strpos("`model'", "/")-1)

display "`model'"

generate _prediction = .

if ("`model'" == "base") {
    * poisson often does not converge, use ppmlhdfe instead
    ppmlhdfe ${outcome} ${size2} ${regressors}
}
if ("`model'" == "size1") {
    * poisson often does not converge, use ppmlhdfe instead
    ppmlhdfe ${outcome} ${regressors}, exposure(${size1})
}
if ("`model'" == "lag") {
    * poisson often does not converge, use ppmlhdfe instead
    ppmlhdfe ${outcome} ${lagged_outcome} ${size2} ${regressors}
}
if ("`model'" == "seasonal") {
    * string variables cannot be used as factors
    egen onum = group(${origin})
    egen dnum = group(${destination})

    ppmlhdfe ${outcome} ${lagged_outcome} ${size2} ${regressors} summer_winter fall_spring
}
if ("`model'" == "fe") {
    * string variables cannot be used as factors
    egen onum = group(${origin})
    egen dnum = group(${destination})

    ppmlhdfe ${outcome} ${size2} ${regressors}, absorb(onum#dnum) d
    egen fe = max(_ppmlhdfe_d), by(onum dnum)
    replace _ppmlhdfe_d = fe if missing(_ppmlhdfe_d)
    
    * if there is no variation in outcome, that is our forecast
    egen min_outcome = min(${outcome}), by(onum dnum)
    egen max_outcome = min(${outcome}), by(onum dnum) 
    replace _prediction = min_outcome if min_outcome == max_outcome
}

summarize ${time}
keep if ${time} == r(max)
replace ${time} = ${time} + ${dt}
predict prediction, mu
* if there is no variation in outcome, that is our forecast
replace prediction = _prediction if !missing(_prediction) & missing(prediction)
replace prediction = int(prediction)

keep ${origin} ${destination} ${time} prediction
sort ${origin} ${destination} ${time}
export delimited "`outf'", replace
