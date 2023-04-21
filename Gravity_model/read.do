args infile outfile
import delimited "`infile'", case(preserve) encoding("utf-8") clear

save "`outfile'", replace