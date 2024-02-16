args outf
local data = substr("`outf'", strpos("`outf'", "/")+1, .)
local data = substr("`data'", 1, strpos("`data'", "/")-1)

display "`data'"

if ("`data'" == "Yearly_trade_data_prediction") {
    local logarithm gdp_o gdp_d total_population_o total_population_d dist 
    local dummies landlocked_o landlocked_d contig comlang_off comcol

    foreach X in `logarithm' {
        generate ln_`X' = ln(`X')
    }
    rename Period year
    rename Value import

    keep iso_? year import ln_* `dummies'
    egen i = group(iso_o iso_d)
    xtset i year
    generate Fimport = F.import
    generate lnGDPtotal = ln_gdp_d + ln_gdp_o
    generate ln_import = ln(import)
    generate zero_import = (import == 0)
    * we cannot use ln_import if import is zero. the zero_import dummy will capture any deviation of ln(1)
    replace ln_import = 0 if zero_import == 1

    * save variable names for future regressions
    global outcome Fimport
    global regressors ln_dist contig comlang_off comcol
    global lagged_outcome ln_import zero_import 
    global origin iso_o
    global destination iso_d
    global size1 lnGDPtotal
    global size2 ln_gdp_d ln_gdp_o
    global time year
    global dt 1
}

if ("`data'" == "GeoDS_mobility_flow_prediction") {
    * FIXME: Total_area_? is a string variable

    local logarithm all_pop_flows_to_d o_pop_flows_to_all population_2019_o	main_roads_length_o population_2019_d	main_roads_length_d distances
    local dummies neighbouring

    * this is wrong, as Timeline is weekly, not yearly
    * however, evaluate.py expects a year variable
    rename Timeline year

    * there are some 0 distances in the data. replace these with 98 miles, the smallest nonzero distance
    replace distances = 98 if distances == 0
    foreach X in `logarithm' {
        generate ln_`X' = ln(`X')
    }
    generate double summer_winter = sin(2 * _pi * (year / 52))
    generate double fall_spring = sin(2 * _pi * ((year + 13) / 52))

    * save variable names for future regressions
    global outcome Fpop_flows
    global regressors ln_population_2019_o ln_population_2019_d ln_distances neighbouring
    global lagged_outcome ln_flow zero_flow 
    global origin origin
    global destination destination
    global size1 lnSIZEtotal
    global size2 ln_all_pop_flows_to_d ln_o_pop_flows_to_all
    global time year
    global dt 1

    egen i = group(${origin} ${destination})
    xtset i ${time}
    generate Fpop_flows = F.pop_flows
    generate lnSIZEtotal = ln_all_pop_flows_to_d + ln_o_pop_flows_to_all
    generate ln_flow = ln(pop_flows)
    generate zero_flow = (pop_flows == 0)
    * we cannot use ln_import if import is zero. the zero_import dummy will capture any deviation of ln(1)
    replace ln_flow = 0 if zero_flow == 1
}

compress
