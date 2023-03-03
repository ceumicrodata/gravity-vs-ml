STATA := stata-mp -b

Gravity_model/prediction.csv: Gravity_model/estimate_poisson.do temp/trade_analysis.dta
	$(STATA) $<
temp/trade_analysis.dta: Gravity_model/select_features.do temp/trade_sample.dta
	$(STATA) $<
temp/trade_sample.dta: Gravity_model/create_sample.do temp/trade_nodelist.dta temp/trade_edgelist.dta
	$(STATA) $<
temp/trade_%list.dta: Gravity_model/read_%s.do temp/trade_%list.csv
	$(STATA) $<