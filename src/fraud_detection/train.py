# TRAINING
# run from the repo root (config.yaml and data/ are read relative to it)
#
# best models end up in the 'best_models' experiment, one run per section:
#   tuning: xgb/iso fit on X_train, meta fit on its oof features (scored on X_val)
#   final:  xgb/iso fit on X, meta fit on its oof features (scored on X_test)
# each run holds the models as 'xgb', 'iso', 'meta' plus the scalers, e.g.
#   mlflow.xgboost.load_model(f'runs:/{run_id}/xgb')

from functools import partial

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from fraud_detection.base_learner import *
from fraud_detection.meta_learner import *

with open('config.yaml') as f:
    config = yaml.safe_load(f)

N_SPLITS = config['n_splits']
N_TUNE = config['n_tune']

sgkf = StratifiedKFold(n_splits = N_SPLITS, shuffle = False)


# logs one section's best models and scalers into a single named run
def log_best_models(run_name, xgb_model, iso_model, meta_model, metric_name, metric, final = False):
    suffix = '_FINAL' if final else ''

    mlflow.set_experiment('best_models')
    with mlflow.start_run(run_name = run_name) as _:
        mlflow.log_params({f'xgb_{k}': v for k, v in xgb_best_param.items()})
        mlflow.log_params({f'iso_{k}': v for k, v in iso_best_param.items()})
        mlflow.log_params({f'meta_{k}': v for k, v in meta_best_param.items()})
        mlflow.log_metric(metric_name, metric)

        mlflow.xgboost.log_model(xgb_model, name = 'xgb')
        mlflow.sklearn.log_model(iso_model, name = 'iso')
        mlflow.sklearn.log_model(meta_model, name = 'meta')

        # meta features have to be scaled the same way at inference
        mlflow.log_artifact(str(scaler_path / f'xgb_scaler{suffix}.pkl'), artifact_path = 'scalers')
        mlflow.log_artifact(str(scaler_path / f'iso_scaler{suffix}.pkl'), artifact_path = 'scalers')


# =========== TUNING RUN (X_train, validated on X_val) ============

X_train = pd.read_csv('data/splits/X_train.csv')
y_train = pd.read_csv('data/splits/y_train.csv')['Class']
X_val = pd.read_csv('data/splits/X_val.csv')
y_val = pd.read_csv('data/splits/y_val.csv')['Class']

# autolog only during the studies, so trial runs get their params/models
# but the best-model fits don't scatter unnamed runs across experiments
mlflow.autolog()

# ===== TUNE BASE LEARNERS =====
x_objective = partial(xgb_objective, sgkf=sgkf, X_train=X_train, y_train=y_train)
i_objective = partial(iso_objective, sgkf=sgkf, X_train=X_train, y_train=y_train)

xgb_best_param, xgb_best_trial = train_tune('xgb_hyperparam', x_objective, N_TUNE)
iso_best_param, iso_best_trial = train_tune('iso_hyperparam', i_objective, N_TUNE)

mlflow.autolog(disable = True)

# ===== META TRAINING FEATURES =====
# oof preds of the best trials
xgb_oof = pd.read_csv(oof_path / 'xgb' / f'oof_trial_{xgb_best_trial}.csv')
iso_oof = pd.read_csv(oof_path / 'iso' / f'oof_trial_{iso_best_trial}.csv')

meta_feat, meta_label = meta_train_features(xgb_oof, iso_oof)

# ===== META VALIDATION FEATURES =====
# best base learners fit on all of X_train, then scored on X_val
xgb_best = train('xgb', X_train, y_train, xgb_best_param)
iso_best = train('iso', X_train, y_train, iso_best_param)

meta_val = meta_val_features(X_val, y_val, xgb_best, iso_best)

# ===== TUNE META LEARNER =====
mlflow.autolog()

m_objective = partial(meta_objective, concat_data=meta_feat, concat_labels=meta_label, meta_val=meta_val, y_val=y_val)
meta_best_param, _ = train_tune('meta_hyperparam', m_objective, N_TUNE)

mlflow.autolog(disable = True)

# ===== LOG BEST TUNING MODELS =====
meta_best = train('logreg', meta_feat, meta_label, meta_best_param)
val_aucpr = average_precision_score(y_val, meta_best.predict_proba(meta_val)[:,1])

log_best_models('tuning', xgb_best, iso_best, meta_best, 'val-aucpr', val_aucpr)


# =========== FINAL RUN (X, tested on X_test) ============
# reuses the hyperparams tuned above

X = pd.read_csv('data/splits/X.csv')
y = pd.read_csv('data/splits/y.csv')['Class']
X_test = pd.read_csv('data/splits/X_test.csv')
y_test = pd.read_csv('data/splits/y_test.csv')['Class']

# ===== META TRAINING FEATURES =====
xgb_oof_final = oof_scores('xgb', sgkf, X, y, xgb_best_param)
iso_oof_final = oof_scores('iso', sgkf, X, y, iso_best_param)

meta_feat_final, meta_label_final = meta_train_features(xgb_oof_final, iso_oof_final, final=True)

# ===== META TEST FEATURES =====
xgb_final = train('xgb', X, y, xgb_best_param)
iso_final = train('iso', X, y, iso_best_param)

meta_test = meta_val_features(X_test, y_test, xgb_final, iso_final, final=True)

# ===== FINAL META LEARNER =====
meta_final = train('logreg', meta_feat_final, meta_label_final, meta_best_param)

# ===== TEST SCORE =====
test_aucpr = average_precision_score(y_test, meta_final.predict_proba(meta_test)[:,1])

log_best_models('final', xgb_final, iso_final, meta_final, 'test-aucpr', test_aucpr, final=True)

print(f'Validation AUCPR: {val_aucpr}')
print(f'Final model AUCPR on test data: {test_aucpr}')
