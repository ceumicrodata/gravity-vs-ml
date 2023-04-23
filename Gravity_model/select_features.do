local logarithm gdp_o gdp_d total_population_o total_population_d dist 
local dummies landlocked_o landlocked_d contig comlang_off comcol

foreach X in `logarithm' {
    generate ln_`X' = ln(`X')
}
rename Period year
rename Value import

keep iso_? year import ln_* `dummies'
compress
