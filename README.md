# Replication Package for: 
## 2022-11-10


## Authors
- Kiss, Olivér
- [Koren, Miklós](https://koren.mk/)
- Ruzicska, György

# Data availability and provenance statements
### Statement about rights
The author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript.

### Summary of availability
All data are publicly available.

### Details on each data source
#### UN Comtrade

> clarify license terms

#### CEPII GeoDist
CEPII GeoDist (CEPII 2011 and Mayer and Zignago 2011) is available via an [Open License 2.0](https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf). We are sharing the data here in compliance with the license terms.

#### World Development Indicators
The World Development Indicators (World Bank 2022) is available with a [CC-BY-4.0 license](https://datacatalog.worldbank.org/search/dataset/0037712). We are sharing the data here in compliance with the license terms.

# Description of programs/code

The entire data cleaning and analysis can be executed with `make`. See the `Makefile` in the root of the folder.

### (Optional, but recommended) License for Code
The code is licensed under a MIT/BSD/GPL/Creative Commons license. See [LICENSE.txt](LICENSE.txt) for details.


# Computational requirements

### Software requirements

<!---
List all of the software requirements, up to and including any operating system requirements, for the entire set of code. It is suggested to distribute most dependencies together with the replication package if allowed, in particular if sourced from unversioned code repositories, Github repos, and personal webpages. In all cases, list the version *you* used. 
-->

- Stata (code was last run with version 17)
  - `here` (can be installed with `net install here, from("https://raw.githubusercontent.com/korenmiklos/here/master/")`)
  - `reghdfe` (can be installed with `ssc install reghdfe`)
- GNU Make

### Memory and runtime requirements
Creating the data extract (`temp/analysis_sample_dyadic.dta`) takes about 15 minutes on a quad-core 3.8GHz machine with 8GB of memory. Each estimation runs in a few minutes.

#### Summary

Approximate time needed to reproduce the analyses on a standard (2020) desktop machine: 30 minutes.


# List of tables and figures

<!---
Your programs should clearly identify the tables and figures as they appear in the manuscript, by number. Sometimes, this may be obvious, e.g. a program called "`table1.do`" generates a file called `table1.png`. Sometimes, mnemonics are used, and a mapping is necessary. In all circumstances, provide a list of tables and figures, identifying the program (and possibly the line number) where a figure is created.

If the public repository is incomplete, because not all data can be provided, as described in the data section, then the list of tables should clearly indicate which tables, figures, and in-text numbers can be reproduced with the public material provided.
-->
# References
- CEPII. 2011. "GeoDist [data set]" Available athttp://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6 Last accessed YYYY-MM-DD. 
- Mayer, T. and Zignago, S. 2011. "Notes on CEPII’s distances measures: the GeoDist Database." CEPII Working Paper 2011-25
- United Nations Statistics Division. 2022. "UN Comtrade [data set]" Available at https://comtrade.un.org/. Last accessed YYYY-MM-DD.
- World Bank. 2022. "World Development Indicators [data set]" Available at https://datacatalog.worldbank.org/search/dataset/0037712 Last accessed YYYY-MM-DD
