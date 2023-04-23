DATA := Output_datasets/Yearly_trade_data_prediction/Chunked_merged_data
CHUNKS := $(patsubst $(DATA)/%.csv,%,$(wildcard $(DATA)/????-????.csv))
STATA := stata -b

.INTERMEDIATE: pred_%.csv

Gravity_model/prediction.csv: $(foreach chunk,$(CHUNKS),Gravity_model/pred_$(chunk).csv)
	head -n1 $< > $@
	tail -n+2 -q $^ >> $@
Gravity_model/pred_%.csv: Gravity_model/estimate_poisson.do $(DATA)/%.csv
	$(STATA) $^ $@
