args infile outfile
import delimited "`infile'", case(preserve) encoding("utf-8") clear

duplicates drop cnum year, force
keep iso_numeric country year gdp area total_population dis_int landlocked

save "`outfile'", replace