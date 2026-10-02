from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import yaml
from sklearn.model_selection import StratifiedKFold
import os
import joblib
from trainer_funcs import *
import mlflow

mlflow.autolog()


# load configs
with open('config.yaml') as f:
    config = yaml.safe_load(f)

N_SPLITS = config['n_splits']
oof_path = config['oof_path']

# load results

iso_table = pd.read_csv('results/iso.csv')
xgb_table = pd.read_csv('results/xgb.csv')
meta_table = pd.read_csv('results/log_reg.csv')

# load split data
X_train = pd.read_csv('data/splits/X.csv')
y_train = pd.read_csv('data/splits/y.csv').squeeze("columns")

X_test = pd.read_csv('data/splits/X_test.csv')
y_test = pd.read_csv('data/splits/y_test.csv').squeeze("columns")

# get params for best runs
xgb_params = dict(xgb_table[[c for c in xgb_table.columns if c.startswith('params.')]].dropna(axis=1).iloc[0])
xgb_params = {k.removeprefix('params.'):v for k,v in xgb_params.items()}
xgb_params['objective'] = 'binary:logitraw'
xgb_params['eval_metric'] = 'aucpr'
xgb_params['seed'] = config['seed']
xgb_params = {k:int(v) if type(v) != str and v.is_integer() else v for k,v in xgb_params.items()}


iso_params = dict(iso_table[[c for c in iso_table.columns if c.startswith('params.')]].dropna(axis=1).iloc[0])
iso_params = {k.removeprefix('params.'):v for k,v in iso_params.items()}
iso_params['random_state'] = config['seed']
iso_params = {k:int(v) if type(v) != str and v.is_integer() else v for k,v in iso_params.items()}


meta_params = dict(meta_table[[c for c in meta_table.columns if c.startswith('params.')]].dropna(axis=1).iloc[0])
meta_params = {k.removeprefix('params.'):v for k,v in meta_params.items()}


###### load params into models and train
sgkf = StratifiedKFold(n_splits = N_SPLITS, shuffle = False)

# isolation forest training: modified version of objective function
iso_oof_preds = np.zeros(len(X_train))

for i, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train)):
            
    train_feat_curr = X_train.iloc[train_index]
    test_feat_curr = X_train.iloc[test_index]
    
    iso_model = IsolationForest(**iso_params).fit(train_feat_curr)
    scores = -iso_model.score_samples(test_feat_curr)
    iso_oof_preds[test_index] = scores

oof_df = pd.DataFrame({'y-train-xgb': y_train, 'iso_oof_preds': iso_oof_preds})

csv_path = f'{oof_path}/iso/oof_trial_FINAL.csv'
oof_df.to_csv(csv_path, index=False)

# XGB training: modified version of objective function
xgb_oof_preds = np.zeros(len(X_train))
for i, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train)):
    train_feat_curr = X_train.iloc[train_index]
    train_label_curr = y_train.iloc[train_index]

    test_feat_curr = X_train.iloc[test_index]
    test_label_curr = y_train.iloc[test_index]

    dtrain= xgb.DMatrix(train_feat_curr, label=train_label_curr)
    dval = xgb.DMatrix(test_feat_curr, label = test_label_curr)
    xgb_model = xgb.train(
                    params = xgb_params,
                    dtrain = dtrain,
                    num_boost_round = 3000,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=200

                )
    val_scores = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration+1)) # output np array
    xgb_oof_preds[test_index] = val_scores

oof_df = pd.DataFrame({'y-train-xgb': y_train, 'xgb_oof_preds': xgb_oof_preds})

csv_path = f'{oof_path}/xgb/oof_trial_FINAL.csv'
oof_df.to_csv(csv_path, index=False)

# construct meta-features

meta_feat_data, meta_feat_labels = meta_train_features(xgb_table, iso_table)

# generate meta-validation features
xgb_best = best_model('xgboost', xgb_table)
iso_best = best_model('sklearn', iso_table)

meta_val = meta_val_features(X_test, y_test, xgb_best, iso_best)

# meta objective
model = LogisticRegression(**meta_params).fit(meta_feat_data, meta_feat_labels)
preds = model.predict_proba(meta_val)[:,1]
aucpr = average_precision_score(y_test, preds)

print(f'Final Model AUCPR on test data: {aucpr}.')

