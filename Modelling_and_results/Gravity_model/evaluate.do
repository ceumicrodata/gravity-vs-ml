args prediction target
tempfile trg

clear
import delimited "`target'", clear varnames(1) case(preserve) encoding(UTF-8)
capture rename Value target
capture rename pop_flows target
capture rename Period year
capture rename Timeline year
capture keep iso_o iso_d year target
capture keep origin destination year target
save `trg', replace

import delimited "`prediction'", clear varnames(1) case(preserve) encoding(UTF-8)
capture merge 1:1 iso_o iso_d year using `trg', keep(master match) nogen
capture merge 1:1 origin destination year using `trg', keep(master match) nogen

capture rename iso_o origin
capture rename iso_d destination

correlate target prediction

* Use ChatGPT 4.0 to translate Oliver's pandas code
generate abs_diff = abs(prediction - target)
summarize abs_diff, meanonly
local mae = r(mean)
display "Mean Absolute Error (MAE): " `mae'

generate sq_diff = (prediction - target)^2
summarize sq_diff, meanonly
local mse = r(mean)
display "Mean Squared Error (MSE): " `mse'

scalar rmse = sqrt(`mse')
display "Root Mean Squared Error (RMSE): " rmse

summarize target, meanonly
local mean_target = r(mean)
local rmae = `mae' / `mean_target'
display "Relative Mean Absolute Error (RMAE): " `rmae'

egen group_mean = mean(target), by(origin destination)
generate sst = (target - `mean_target')^2
generate ssw = (target - group_mean)^2
generate ssr = (target - prediction)^2
egen total_sst = sum(sst)
egen total_ssw = sum(ssw)
egen total_ssr = sum(ssr)
scalar r2 = 1 - (total_ssr/total_sst)
display "R-squared (R2): " r2

scalar within_r2 = 1 - (total_ssr/total_ssw)
display "Within R-squared (R2): " within_r2
