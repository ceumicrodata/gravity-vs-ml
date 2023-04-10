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

for result_file in result_files:
    try:
        predictions = pd.read_csv(result_file).prediction
        targets = pd.read_csv(result_file).target
        measures = dict(zip(measure_names,(mae(predictions,targets), rmae(predictions,targets), mse(predictions, targets), rmse(predictions, targets), r2(predictions, targets), cpc(predictions,targets))))
        measures['sha'] = sha
        measures['path'] = result_file
    except AttributeError:
        warnings.warn(f"Incorrect structure found in {result_file}, nans will be reported")
        measures = dict(zip(measure_names,(np.nan, np.nan, np.nan, np.nan, np.nan)))
        measures['sha'] = sha
        measures['path'] = result_file
    except:
        warnings.warn(f"Unhandled error when processing {result_file}, nans will be reported")
        measures = dict(zip(measure_names, (np.nan, np.nan, np.nan, np.nan, np.nan)))
        measures['sha'] = sha
        measures['path'] = result_file
    if pathlib.Path('../evaluations.csv').is_file():
        pre_existing_results = pd.read_csv("../evaluations.csv")
        results = pd.concat([pre_existing_results, pd.DataFrame([measures])])
    else:
        results = pd.DataFrame([measures])
    results.to_csv('../evaluations.csv', index=False)
