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

The provided code reproduces:
<!---
pick one
-->
- All numbers provided in text in the paper
- All tables and figures in the paper
- Selected tables and figures in the paper, as explained and justified below.


| Figure/Table #    | Program                  | Line Number | Output file                      | Note                            |
|-------------------|--------------------------|-------------|----------------------------------|---------------------------------|
| Table 1           | 02_analysis/table1.do    |             | summarystats.csv                 ||
| Table 2           | 02_analysis/table2and3.do| 15          | table2.csv                       ||
| Table 3           | 02_analysis/table2and3.do| 145         | table3.csv                       ||
| Figure 1          | n.a. (no data)           |             |                                  | Source: Herodus (2011)          |
| Figure 2          | 02_analysis/fig2.do      |             | figure2.png                      ||
| Figure 3          | 02_analysis/fig3.do      |             | figure-robustness.png            | Requires confidential data      |


# References

<!---
As in any scientific manuscript, you should have proper references. For instance, in this sample README, we cited "Ruggles et al, 2019" and "DESE, 2019" in a Data Availability Statement. The reference should thus be listed here, in the style of your journal.
-->

- Steven Ruggles, Steven M. Manson, Tracy A. Kugler, David A. Haynes II, David C. Van Riper, and Maryia Bakhtsiyarava. 2018. "IPUMS Terra: Integrated Data on Population and Environment: Version 2 [dataset]." Minneapolis, MN: *Minnesota Population Center, IPUMS*. https://doi.org/10.18128/D090.V2
- Department of Elementary and Secondary Education (DESE), 2019. "Student outcomes database [dataset]" *Massachusetts Department of Elementary and Secondary Education (DESE)*. Accessed January 15, 2019.
- U.S. Bureau of Economic Analysis (BEA). 2016. “Table 30: "Economic Profile by County, 1969-2016.” (accessed Sept 1, 2017).
- Inglehart, R., C. Haerpfer, A. Moreno, C. Welzel, K. Kizilova, J. Diez-Medrano, M. Lagos, P. Norris, E. Ponarin & B. Puranen et al. (eds.). 2014. World Values Survey: Round Six - Country-Pooled Datafile Version: http://www.worldvaluessurvey.org/WVSDocumentationWV6.jsp. Madrid: JD Systems Institute.
