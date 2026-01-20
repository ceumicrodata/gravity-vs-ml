# Can Machine Learning Beat Gravity in Flow Prediction?

**Replication Package**  
Date: 2022-11-10

## Citation

If you use this code or data in your research, please cite:

> Ruzicska, G., Chariag, R., Kiss, O., Koren, M. (2024). Can Machine Learning Beat Gravity in Flow Prediction?. In: Matyas, L. (eds) The Econometrics of Multi-dimensional Panels. Advanced Studies in Theoretical and Applied Econometrics, vol 54. Springer, Cham. https://doi.org/10.1007/978-3-031-49849-7_16

**DOI**: [10.1007/978-3-031-49849-7_16](https://doi.org/10.1007/978-3-031-49849-7_16)

## Authors
- Chariag, Ramzi
- Kiss, Olivér
- [Koren, Miklós](https://koren.mk/)
- Ruzicska, György

---

## Overview

This repository contains the replication package for a study comparing traditional gravity models with various machine learning approaches for predicting flows between locations. The project evaluates different modeling techniques on three distinct datasets:

1. **GeoDS Mobility Data**: US state-to-state mobility flows
2. **Google Mobility Data**: County-level mobility flows in the US
3. **International Trade Data**: Annual bilateral trade flows between countries

The repository implements and compares multiple modeling approaches:
- **Gravity Models** (Poisson regression with fixed effects)
- **Neural Networks** (Deep learning models)
- **Random Forests**
- **Graph Neural Networks (GNN)** (DCRNN-based models)
- **Ensemble Models**
- **Random Walk Baseline**

---

## Project Structure

```
gravity-vs-ml/
├── Data_extraction/           # Scripts to fetch raw data from various sources
├── Data_cleaning_and_validation/  # Data cleaning and validation scripts
├── Input_datasets/            # Raw and processed input datasets
├── Output_datasets/           # Processed datasets ready for modeling
├── Modelling_and_results/     # Implementation of all modeling approaches
│   ├── Gravity_model/         # Stata-based gravity model implementations
│   ├── Neural_network/        # PyTorch-based neural network models
│   ├── Random_forest/         # Random forest implementations
│   ├── GNN_model/             # Graph neural network models
│   ├── Ensemble/              # Ensemble model combining predictions
│   └── Random_walk/           # Random walk baseline
├── Evaluation/                # Evaluation scripts and metrics
├── Evaluations/               # Evaluation results (CSV files)
└── README.md                  # This file
```

---

## Installation and Setup

### Software Requirements

#### Python Environment
- Python 3.11+
- Dependencies managed via Poetry (see `pyproject.toml`)

To set up the Python environment:
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

Alternatively, if using pip:
```bash
pip install pandas torch scikit-learn networkx jupyter tqdm requests geopy haversine wbgapi ray
```

#### Stata
- Stata version 17 or later
- Required Stata packages:
  - `here` (install with: `net install here, from("https://raw.githubusercontent.com/korenmiklos/here/master/")`)
  - `reghdfe` (install with: `ssc install reghdfe`)

#### Other Requirements
- GNU Make (for running the complete pipeline)

### Memory and Runtime Requirements

- **Minimum**: Quad-core 3.8GHz machine with 8GB RAM
- **Data extraction**: ~15 minutes
- **Model training**: Varies by model (from minutes to hours)
- **Full pipeline**: Approximately 30+ minutes on a standard (2020) desktop machine

---

## Data Sources

### Statement about Rights
The authors have legitimate access to and permission to use all data in this manuscript.

### Summary
All data used in this study are publicly available from the sources listed below.

### Detailed Data Sources

#### UN Comtrade
International trade data. License terms: [To be clarified]

#### CEPII GeoDist
Geographic distance data between countries (CEPII 2011, Mayer and Zignago 2011).  
**License**: [Open License 2.0](https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf)  
**Source**: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6

#### World Development Indicators
Economic indicators from the World Bank (2022).  
**License**: [CC-BY-4.0](https://datacatalog.worldbank.org/search/dataset/0037712)  
**Source**: https://datacatalog.worldbank.org/search/dataset/0037712

#### Country Codes
ISO country code mappings.  
**License**: [PDDL License](https://opendatacommons.org/licenses/pddl/)  
**Source**: [Data hub](https://datahub.io/core/country-codes)

#### Country Groups
Geographic and economic country groupings.  
**License**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)  
**Source**: [GeoDataSource](https://github.com/geodatasource/country-grouping-terminology)

#### US Mobility Data
- **GeoDS Mobility**: State-to-state mobility flows
- **Google Mobility**: County-level mobility data from [Google COVID-19 Mobility Reports](https://www.google.com/covid19/mobility/)
- **County Adjacencies**: From [US Census Bureau](https://www.census.gov/geographies/reference-files/2010/geo/county-adjacency.html)
- **Additional Context**: See [covid-US-spatiotemporal](https://github.com/kiss-oliver/covid-US-spatiotemporal)

#### OpenStreetMap (OSM) Features
Geographic features and road network data for mobility prediction.

---

## Workflow

### 1. Data Extraction (`Data_extraction/`)
Scripts to download raw data from various sources:
- `fetch_cepii_data.py`: Distance and geographic data
- `fetch_trade_data.py`: UN Comtrade trade flows
- `fetch_WBG.py`: World Bank indicators
- `fetch_GeoDS_data.py`: GeoDS mobility data
- `fetch_Google_mobility.py`: Google mobility data
- `fetch_OSM.py`: OpenStreetMap features
- Additional scripts for US state data (population, distances, sizes)

### 2. Data Cleaning and Validation (`Data_cleaning_and_validation/`)
- `GeoDS_compile_and_validate.py`: Clean and validate GeoDS mobility data
- `Google_mobility_compile_and_validate.py`: Clean and validate Google mobility data
- `trade_compile_and_validate.py`: Clean and validate trade data

### 3. Data Processing (`Output_datasets/`)
Processed datasets organized by domain:
- Edge lists (origin-destination pairs)
- Node lists (location features)
- Target lists (ground truth flows)
- Chunked datasets for efficient processing

### 4. Modeling (`Modelling_and_results/`)

#### Gravity Models (`Gravity_model/`)
- Traditional Poisson gravity models with fixed effects
- Implemented in Stata
- Variants: base, fixed effects, lagged variables
- Run using: `make` (see `Makefile`)

#### Neural Networks (`Neural_network/`)
- Deep learning models implemented in PyTorch
- Configurations: full feature set, limited features, with/without lagged variables
- Hyperparameter tuning with Ray Tune
- Results saved in domain-specific subdirectories

#### Random Forests (`Random_forest/`)
- Random forest models with various feature configurations
- Implemented using scikit-learn
- Scripts: `RF_[dataset]_[config].py`

#### Graph Neural Networks (`GNN_model/`)
- DCRNN (Diffusion Convolutional Recurrent Neural Network) models
- Graph-based learning for spatial flow prediction
- Jupyter notebooks for exploration: `gnn.ipynb`, `gnn_trade.ipynb`

#### Ensemble Models (`Ensemble/`)
- Combines predictions from multiple models
- Weighted averaging of model outputs
- `ensemble_model.py` for generating final predictions

#### Random Walk Baseline (`Random_walk/`)
- Simple baseline model for comparison
- Implemented in Stata

### 5. Evaluation (`Evaluation/`)
The `evaluate.py` script computes multiple evaluation metrics:
- **R² (R-squared)**: Overall fit
- **Within R²**: Fit within origin-destination pairs
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMAE (Relative Mean Absolute Error)**: Normalized MAE
- **Bias**: Average prediction bias
- **CPC (Common Part of Commuters)**: Overlap measure for mobility flows

Results are saved in:
- `Evaluations/all_runs/`: Historical results across all runs
- `Evaluations/most_recent_run/`: Results from the latest evaluation

---

## Running the Code

### Quick Start

To run the entire pipeline:
```bash
make
```

This will execute data processing and gravity model estimation (where applicable).

### Individual Components

#### Run Data Extraction
```bash
cd Data_extraction
python fetch_cepii_data.py
python fetch_trade_data.py
# ... (run other extraction scripts as needed)
```

#### Run Data Cleaning
```bash
cd Data_cleaning_and_validation
python GeoDS_compile_and_validate.py
python Google_mobility_compile_and_validate.py
python trade_compile_and_validate.py
```

#### Train Models

**Neural Networks:**
```bash
cd Modelling_and_results/Neural_network/neural_network_gravity
python main.py GeoDS_full
python main.py trade
# ... (other configurations)
```

**Random Forests:**
```bash
cd Modelling_and_results/Random_forest
python RF_GeoDS.py
python RF_trade.py
# ... (other scripts)
```

**Gravity Models:**
The Makefile handles gravity model estimation automatically when you run `make`.

#### Evaluate Results
```bash
cd Evaluation
python evaluate.py
```

This will scan all prediction files and compute evaluation metrics, saving results to `Evaluations/`.

---

## Datasets

### Trade Data (`Yearly_trade_data_prediction/`)
- **Scope**: Bilateral trade flows between countries
- **Time Period**: 2000-2019 (20 years)
- **Observations**: 146,200 origin-destination-year triplets
- **Features**: Distance, economic indicators, country characteristics

### GeoDS Mobility (`GeoDS_mobility_flow_prediction/`)
- **Scope**: US state-to-state mobility flows
- **Time Period**: Years 50-150 (11 time periods)
- **Observations**: 24,816 origin-destination-year triplets
- **Features**: Geographic distance, population, state characteristics, road networks

### Google Mobility (`Google_mobility_flow_prediction/`)
- **Scope**: County-level mobility flows in the US
- **Time Period**: Years 350-950 (13 time periods)
- **Observations**: 3,744 origin-destination-year triplets
- **Features**: Mobility characteristics, county adjacencies, demographic data

---

## License

The code in this repository is licensed under **CC0 1.0 Universal** (Public Domain Dedication). See [LICENSE](LICENSE) for details.

---

## List of Tables and Figures

> **Note**: This section should be updated to map specific programs and line numbers to tables and figures in the manuscript. Programs that generate tables/figures should be clearly identified here.

---

## References

- Ruzicska, G., Chariag, R., Kiss, O., Koren, M. (2024). Can Machine Learning Beat Gravity in Flow Prediction?. In: Matyas, L. (eds) The Econometrics of Multi-dimensional Panels. Advanced Studies in Theoretical and Applied Econometrics, vol 54. Springer, Cham. https://doi.org/10.1007/978-3-031-49849-7_16
- CEPII. 2011. "GeoDist [data set]." Available at http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6. Last accessed YYYY-MM-DD.
- Mayer, T. and Zignago, S. 2011. "Notes on CEPII's distances measures: the GeoDist Database." CEPII Working Paper 2011-25.
- United Nations Statistics Division. 2022. "UN Comtrade [data set]." Available at https://comtrade.un.org/. Last accessed YYYY-MM-DD.
- World Bank. 2022. "World Development Indicators [data set]." Available at https://datacatalog.worldbank.org/search/dataset/0037712. Last accessed YYYY-MM-DD.

