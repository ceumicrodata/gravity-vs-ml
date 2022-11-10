# Pre-analysis plan
## Analysis goals
The goal of this project is to evaluate the out-of-sample predictive performance of different econometric models for the volume of trade between pairs of countries. 

## Data sources
Data should cover bilateral trade for a large sample of countries. EU COMEXT? UN Comtrade?

Auxiliary data?

Questions:
1. Aggregate trade volume or product-specific?

## Sample selection criteria
### Training data
Training data should start after the global recession in 2008-2009. 2010--2014

### Testing data
Models will be evaluated out of sample, at 1, 3, and 5-year horizons. Evaluation data is hence for 2015, 2017, and 2019.

## Models
### Reduced-form gravity
### Fixed-effect gravity
### Structural gravity

## Evaluation criteria
Each model should predict a probability of nonzero imports between a pair of countries, $\pi_{ijt}$. This will be evaluated on the testing data using the Area Under the Curve.

Each model should also predict a dollar value for imports between a pair of countries, for each pair for which $\pi_{ijt}>0$. These will be evaluated only on country pairs for which the actual imports were positive. The evaluation criteria is root mean squared error for *log* imports,
$$
\sigma(\ln M_{ijt} - \ln \hat M_{ijt}).
$$

### Conditional forecasts
Models may use future data on GDP to construct conditional forecasts.