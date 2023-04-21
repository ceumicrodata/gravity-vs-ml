DATA := Output_datasets/Yearly_trade_data_prediction/Chunked_merged_data
CHUNKS := $(patsubst $(DATA)/%.csv,%,$(wildcard $(DATA)/????-????.csv))
STATA := stata -b

all: $(foreach chunk,$(CHUNKS),Gravity_model/prediction_$(chunk).csv)
Gravity_model/prediction_%.csv: Gravity_model/estimate_poisson.do temp/trade_%.dta
	$(STATA) $^ $@
temp/trade_%.dta: Gravity_model/read.do $(DATA)/%.csv
	$(STATA) $^ $@