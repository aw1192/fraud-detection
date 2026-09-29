# TRAINING

from functools import partial

import mlflow
import yaml
from sklearn.model_selection import StratifiedKFold
import os

from trainer_funcs import *
import joblib

with open('config.yaml') as f:
    config = yaml.safe_load(f)

mlflow.autolog()

mlflow.get_tracking_uri()

SEED = config['seed']
N_SPLITS = config['n_splits']
N_TUNE = config['n_tune']
TRIAL = config['trial']

os.makedirs('oof_scores/iso', exist_ok=True)
os.makedirs('oof_scores/xgb', exist_ok=True)
os.makedirs('results', exist_ok=True)

X_train = pd.read_csv('data/splits/X_train.csv')
y_train = pd.read_csv('data/splits/y_train.csv')['Class']
X_val = pd.read_csv('data/splits/X_val.csv')
y_val = pd.read_csv('data/splits/y_val.csv')['Class']

# else: # used to train final model once optimal hyperparameters have been found
#     X_train = pd.read_csv('data/splits/X.csv')
#     y_train = pd.read_csv('data/splits/y.csv')
#     X_val = pd.read_csv('data/splits/X_test.csv') # this is actually the test set but named val for simplicity
#     y_val = pd.read_csv('data/splits/y_test.csv')


# setup
sgkf = StratifiedKFold(n_splits = N_SPLITS, shuffle = False)
x_objective = partial(xgb_objective, sgkf=sgkf, X_train=X_train, y_train=y_train)
i_objective = partial(iso_objective, sgkf=sgkf, X_train=X_train, y_train=y_train)

# training
xgb_table = train_tune('xgb_hyperparam', x_objective, N_TUNE, final=False)
iso_table = train_tune('iso_hyperparam', i_objective, N_TUNE, final=False)

# loading best base learners
xgb_best = best_model('xgboost', xgb_table)
iso_best = best_model('sklearn', iso_table)

xgb_table.to_csv('results/xgb.csv')
iso_table.to_csv('results/iso.csv')

# metafeatures for metalearner
meta_feat, meta_label = meta_train_features(xgb_table, iso_table)

# meta-validation data
meta_val = meta_val_features(X_val, y_val, xgb_best, iso_best)

# fit meta learner
m_objective = partial(meta_objective, concat_data=meta_feat, concat_labels=meta_label, meta_val=meta_val, y_val=y_val)
meta_table = train_tune('meta_hyperparam', m_objective, N_TUNE, final=True)

meta_table.to_csv('results/log_reg.csv')