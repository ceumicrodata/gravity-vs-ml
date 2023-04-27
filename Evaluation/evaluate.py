import pathlib
import numpy as np
import pandas as pd
import warnings
import git


def mae(predictions: pd.Series, targets: pd.Series) -> float:
    return float(np.mean(np.abs(predictions-targets)))


def mse(predictions: pd.Series, targets: pd.Series) -> float:
    return float(np.mean((predictions-targets)**2))


def rmse(predictions: pd.Series, targets: pd.Series) -> float:
    return np.sqrt(mse(predictions,targets))


def rmae(predictions: pd.Series, targets: pd.Series) -> float:
    return mae(predictions,targets)/np.mean(targets)


def r2(predictions: pd.Series, targets: pd.Series) -> float:
    y_mean = np.mean(targets)
    sst = np.sum((targets-y_mean)**2)
    ssr = np.sum((targets-predictions)**2)
    return 1-ssr/sst

def cpc(predictions: pd.Series, targets: pd.Series) -> float:
    numerator = 2.0 * np.sum(np.minimum(predictions, targets)) 
    denominator = np.sum(predictions) + np.sum(targets)
    return numerator/denominator

project_home = pathlib.Path("../")
result_files = list(str(x) for x in project_home.rglob('*') if 'prediction' in str(x).split("/")[-1] and '.csv' in str(x))
print(f"Found {len(result_files)} model result file(s):\n",*[f"{x}\n" for x in result_files])

measure_names = ["MAE", "RMAE", "MSE", "RMSE", "PSEUDOR2", "CommonPartOfCommuters"]
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

# Separate result files
google = [f for f in result_files if 'google' in f.lower()]
geods = [f for f in result_files if 'geods' in f.lower()]
trade = [f for f in result_files if 'trade' in f.lower()]

# Evaluate trade result files
trade_data = pd.read_csv("../Output_datasets/Yearly_trade_data_prediction/trade_edgelist.csv")
trade_data = trade_data.rename(columns={"Value":"target", "Period":"year"})
for result_file in trade:
    print(f'Processing {result_file}')
    results = pd.read_csv(result_file)
    if 'year' not in results or 'iso_d' not in results or 'iso_o' not in results or 'prediction' not in results:
        warnings.warn(f'{result_file} has missing columns, skipping.')
        continue
    if not results.shape[0]==146200:
        warnings.warn(f'{result_file} has incorrect number of records, skipping.')
        continue
    if not results.year.nunique()==20 or results.year.astype(int).min()!=2000 or results.year.astype(int).max()!=2019 or results.year.value_counts().nunique()!=1:
        warnings.warn(f'{result_file} has incorrect years')
        continue
    try:
        results = results[[col for col in results if col!='target']].merge(trade_data[['year', 'iso_o', 'iso_d', 'target']], how='left')
        assert results.shape[0]==146200
        predictions = results.prediction
        targets = results.target
        measures = dict(zip(measure_names,(mae(predictions,targets), rmae(predictions,targets), mse(predictions, targets), rmse(predictions, targets), r2(predictions, targets), cpc(predictions,targets))))
        measures['sha'] = sha
        measures['path'] = result_file
    except:
        warnings.warn(f"Unhandled error when processing {result_file}, skipping")
        continue
    if pathlib.Path('../evaluations.csv').is_file():
        pre_existing_results = pd.read_csv("../evaluations.csv")
        results = pd.concat([pre_existing_results, pd.DataFrame([measures])])
    else:
        results = pd.DataFrame([measures])
    results.to_csv('../evaluations.csv', index=False)
