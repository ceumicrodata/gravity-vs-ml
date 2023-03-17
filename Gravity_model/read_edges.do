args infile outfile
import delimited "`infile'", case(preserve) encoding("utf-8") clear

rename Period year
rename Value import
drop v1

save "`outfile'", replace