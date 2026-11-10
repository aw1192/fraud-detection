import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

with open("config.yaml") as f:
    config = yaml.safe_load(f)

SEED = config['seed']
scaler_path = Path(__file__).resolve().parents[2] / 'artifacts' / 'scalers'
scaler_path.mkdir(parents=True, exist_ok=True)

# GET THE METAFEATURES OF TRAINING SET
def meta_train_features(xgb_oof, iso_oof, final = False):
    '''
    xgb_oof, iso_oof: dataframes of the best trials' oof predictions (loaded from oof_trial_{n}.csv)
    '''

    # concatenate OOF dataframes and clean
    concat = pd.concat([xgb_oof, iso_oof], axis=1)
    concat = concat.drop(columns = 'y-train-iso')
    concat.rename(columns = {'y-train-xgb':'y-train'}, inplace = True)

    # ===== SCALERS ======
    iso_scaler = StandardScaler()
    xgb_scaler = StandardScaler()

    # removes the base 2 exponential
    log_iso = -np.log2(np.array(concat['iso_oof_preds']).reshape(-1,1)) # outputs E[h(x)]/c(n)

    # fit the scaler, then apply to the data
    concat['iso_oof_preds'] = iso_scaler.fit_transform(log_iso)  
    concat['xgb_oof_preds'] = xgb_scaler.fit_transform(np.array(concat['xgb_oof_preds']).reshape(-1,1))

    # separate the df into the data and the labels
    meta_feat_data = concat[['xgb_oof_preds', 'iso_oof_preds']]
    meta_feat_labels = concat['y-train']

    os.makedirs(scaler_path, exist_ok = True)

    # save scalers; this is used to scale the validation data with the training scales
    if final:
        joblib.dump(xgb_scaler, scaler_path / 'xgb_scaler_FINAL.pkl')
        joblib.dump(iso_scaler, scaler_path / 'iso_scaler_FINAL.pkl')
    else:

        joblib.dump(xgb_scaler, scaler_path / 'xgb_scaler.pkl')
        joblib.dump(iso_scaler, scaler_path / 'iso_scaler.pkl')

    return meta_feat_data, meta_feat_labels


def meta_val_features(X_val, y_val, xgb_best, iso_best, final = False):
    '''
    X_val, y_val: the validation/test data as a df
    xgb_best, iso_best: the best models of each type
    '''

    # load the scalers
    if final:
        xgb_scaler = joblib.load(scaler_path / 'xgb_scaler_FINAL.pkl')
        iso_scaler = joblib.load(scaler_path / 'iso_scaler_FINAL.pkl')
    else:
        xgb_scaler = joblib.load(scaler_path / 'xgb_scaler.pkl')
        iso_scaler = joblib.load(scaler_path / 'iso_scaler.pkl')

    # make into DMatrix for xgb
    dval= xgb.DMatrix(X_val, label = y_val)

    # get meta validation features by predicting using the best base models
    meta_xgb = xgb_best.predict(dval)
    meta_iso = -iso_best.score_samples(X_val)

    # transform the validation data
    log_iso_val = -np.log2(np.array(meta_iso).reshape(-1, 1))  

    meta_val = pd.DataFrame({
        'xgb_oof_preds': xgb_scaler.transform(np.array(meta_xgb).reshape(-1, 1)).ravel(),
        'iso_oof_preds': iso_scaler.transform(log_iso_val).ravel(),
    })    

    return meta_val

# META-LEARNER OBJECTIVE

def meta_objective(trial, concat_data, concat_labels, meta_val, y_val):

    # ===== START RUN =====
    with mlflow.start_run(nested=True, run_name=f'meta_trial_{trial.number}') as _:

        # fit the model
        model = LogisticRegression(random_state = SEED,
                               penalty = 'l2',
                               C = trial.suggest_float('C', 0.5, 10)
        ).fit(concat_data, concat_labels)

        # evaluate
        preds = model.predict_proba(meta_val)[:,1]
        aucpr = average_precision_score(y_val, preds)
        mlflow.log_metric('aucpr', aucpr)
        
        return aucpr  