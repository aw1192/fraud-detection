# FULL TRAIN/TUNE
import os
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

with open("config.yaml") as f:
    config = yaml.safe_load(f)

SEED = config['seed']
oof_path = Path(__file__).resolve().parents[2] / 'artifacts' / 'oof_scores'
oof_path.mkdir(parents=True, exist_ok=True)

# ==================== ISOLATION FOREST OBJECTIVE FUNCTION ============================

def iso_objective(trial, sgkf, X_train, y_train):
    

    # ===== START RUN =====

    with mlflow.start_run(nested=True, run_name=f'iso_trial_{trial.number}') as _:
        
        mlflow.set_tag("trial-number", trial.number)

        # setup score tracking
        oof_preds = np.zeros(len(X_train))
        aucpr_scores = []
        
        for _, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train)):

            # get the indices of the train and val batch
            train_feat_curr = X_train.iloc[train_index]
            test_feat_curr = X_train.iloc[test_index]

            # ====== TRAIN FOREST ======
            model = IsolationForest(random_state = SEED,
                                    n_estimators = trial.suggest_int('n_estimators', 75, 300),
                                    max_samples = trial.suggest_int('max_samples', 50, 500),
                                    max_features = trial.suggest_float('max_features', 0.1, 1)                                
            ).fit(train_feat_curr)

            # ==== VALIDATION ====

            # since sklearn's iso forest is negative, negate it back to positive
            scores = -model.score_samples(test_feat_curr)
            oof_preds[test_index] = scores

            # get the scores
            y_val_curr = y_train.iloc[test_index]
            aucpr = average_precision_score(y_val_curr, scores)
            aucpr_scores.append(aucpr)

        # compute mean aucpr across all folds
        mean_aucpr = sum(aucpr_scores)/len(aucpr_scores)
        

        # ======= OUTPUTS ========
        mlflow.log_metric('mean-aucpr', mean_aucpr)

        # save oof preds for the meta learner
        oof_df = pd.DataFrame({'y-train-iso': y_train, 'iso_oof_preds': oof_preds})

        os.makedirs(oof_path / 'iso', exist_ok=True)
        csv_path = oof_path / 'iso' / f'oof_trial_{trial.number}.csv'
        oof_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        return mean_aucpr



# ==================== XGBOOST TREE OBJECTIVE FUNCTION ============================


def xgb_objective(trial, sgkf, X_train, y_train): 
    legit_over_fraud = ((len(y_train)-sum(y_train))/sum(y_train))

    # ======== HYPERPARAMETERS =========

    objective = 'binary:logitraw'
    max_rounds = 3000
    early_stopping_rounds = 200


    scale_pos_weight = trial.suggest_float('scale_pos_weight', 0.2*legit_over_fraud, 5 * legit_over_fraud, log=True) 
    eta = trial.suggest_float('eta', 0.01, 0.5, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0.001, 10, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1)
    min_child_weight = trial.suggest_int('min_child_weight', 5, 100)


    # ========= START RUN =========
    with mlflow.start_run(nested=True, run_name=f'xgb_trial_{trial.number}') as _:

        mlflow.set_tag("trial-number", trial.number)

        # hyperparameter setup
        params = {
                'objective': objective,
                'eval_metric': "aucpr",
                'scale_pos_weight': scale_pos_weight,
                'max_depth': max_depth,
                'eta': eta,
                'subsample': subsample,
                'reg_lambda': reg_lambda,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight,
                'seed': SEED
            }

        # ====== SCORING SETUP ======
        evals = {}
        oof_preds = np.zeros(len(X_train))
        aucpr_scores = []
        best_rounds = []

        # ====== TRAINING ======
        for _, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train)):

            # training data
            train_feat_curr = X_train.iloc[train_index]
            train_label_curr = y_train.iloc[train_index]

            # test data
            test_feat_curr = X_train.iloc[test_index]
            test_label_curr = y_train.iloc[test_index]

            # put into DMatrix
            dtrain= xgb.DMatrix(train_feat_curr, label=train_label_curr)
            dval = xgb.DMatrix(test_feat_curr, label = test_label_curr)

            # train model
            bst = xgb.train(
                params = params,
                dtrain = dtrain,
                num_boost_round = max_rounds,
                evals = [(dtrain, 'dtrain'), (dval, 'dval')],
                early_stopping_rounds = early_stopping_rounds,
                evals_result = evals,
                verbose_eval = int(early_stopping_rounds/2)
            )

            # ===== VALIDATION AND SCORING =====
            val_scores = bst.predict(dval, iteration_range=(0, bst.best_iteration+1)) # output np array
            y_val_true = dval.get_label()
            aucpr = average_precision_score(y_val_true, val_scores)

            # put preds in oof_preds and append aucpr scores
            oof_preds[test_index] = val_scores
            aucpr_scores.append(aucpr)
            best_rounds.append(bst.best_iteration + 1)

        # mean aucpr over all folds
        mean_aucpr = sum(aucpr_scores)/len(aucpr_scores)

        # mean early-stopped round count, used as num_boost_round when training on all data
        n_rounds = int(np.mean(best_rounds))
        trial.set_user_attr('n_rounds', n_rounds)

        # ======== OUTPUTS =========
        mlflow.log_metric('mean-aucpr', mean_aucpr)
        mlflow.log_metric('n-rounds', n_rounds)

        # save oof preds for the meta learner
        oof_df = pd.DataFrame({'y-train-xgb': y_train, 'xgb_oof_preds': oof_preds})

        os.makedirs(oof_path / 'xgb', exist_ok=True)
        csv_path = oof_path / 'xgb' / f'oof_trial_{trial.number}.csv'
        oof_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        return mean_aucpr



# ==================== FULL TRAIN-TUNE RUNS =============================
def train_tune(name, objective, n_trials):
    mlflow.set_experiment(f'{name}')

    # ======= START PARENT RUN =======
    with mlflow.start_run(run_name = 'study') as _:
        mlflow.log_param('n_trials', n_trials)

        # create the study, and optimize what is being returned (mean aucpr)
        study = optuna.create_study(direction = 'maximize')

        # unwrap functools.partial so the checks below match the base objective
        base_objective = getattr(objective, 'func', objective)

        # enqueue the default hyperparameters as a first trial
        if base_objective == iso_objective:
            study.enqueue_trial({'n_estimators': 100, 'max_samples': 256, 'max_features': 1.0})

        elif base_objective == xgb_objective:
            study.enqueue_trial({
                'eta': 0.3,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1.0,
            })

        # enqueue default logistic regression hyperparameters
        else:
            study.enqueue_trial({
                'penalty': 'l2',
                'C': 1.0
            })

        # start the study
        study.optimize(objective, n_trials=n_trials)
    
        # ======== LOGGING =========
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_mean_aucpr": study.best_value})

    # ===== GET BEST TRIAL ======
    # user attrs carry non-tuned values the final train needs (e.g. xgb n_rounds)
    # trial number locates the best trial's oof csv (oof_trial_{number}.csv)
    best = study.best_trial
    return {**best.params, **best.user_attrs}, best.number

# ================== FULL TRAIN RUNS ==================
# (used once hyperparameters have been decided)

def train(type: Literal['xgb', 'iso', 'logreg'], X_train, y_train, best_params):
    '''
    best_params: the params dict returned by train_tune
    returns the model fit on all of X_train
    '''

    # copy so the caller's dict isn't mutated by the pop below
    params = dict(best_params)

    if type == 'xgb':
        # setup
        dtrain= xgb.DMatrix(X_train, label=y_train)

        # no val set to early stop on, so use the mean best round from cv
        n_rounds = params.pop('n_rounds')

        # add back the fixed params that optuna didn't tune
        params.update({
            'objective': 'binary:logitraw',
            'eval_metric': 'aucpr',
            'seed': SEED
        })

        # train model
        model = xgb.train(
            params = params,
            dtrain = dtrain,
            num_boost_round = n_rounds,
            evals = [(dtrain, 'dtrain')],
            verbose_eval = 100
        )

    elif type == 'iso':
        # unsupervised, labels aren't used
        model = IsolationForest(random_state = SEED, **params).fit(X_train)

    else:
        model = LogisticRegression(random_state = SEED, penalty = 'l2', **params).fit(X_train, y_train)

    return model

# ================== OOF SCORES WITH FIXED PARAMS ==================
# (same cv as the objectives, but with already-tuned params; used for the final meta features)

def oof_scores(type: Literal['xgb', 'iso'], sgkf, X_train, y_train, best_params):
    '''
    best_params: the params dict returned by train_tune
    returns the oof df in the same format as the objectives' csvs, and saves it as oof_trial_FINAL.csv
    '''

    params = dict(best_params)
    oof_preds = np.zeros(len(X_train))

    if type == 'xgb':
        # each fold early stops on its own val fold, like in tuning
        params.pop('n_rounds', None)
        params.update({
            'objective': 'binary:logitraw',
            'eval_metric': 'aucpr',
            'seed': SEED
        })

    for _, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train)):

        train_feat_curr = X_train.iloc[train_index]
        test_feat_curr = X_train.iloc[test_index]

        if type == 'xgb':
            dtrain = xgb.DMatrix(train_feat_curr, label=y_train.iloc[train_index])
            dval = xgb.DMatrix(test_feat_curr, label=y_train.iloc[test_index])

            bst = xgb.train(
                params = params,
                dtrain = dtrain,
                num_boost_round = 3000,
                evals = [(dval, 'dval')],
                early_stopping_rounds = 200,
                verbose_eval = 100
            )
            oof_preds[test_index] = bst.predict(dval, iteration_range=(0, bst.best_iteration+1))

        else:
            model = IsolationForest(random_state = SEED, **params).fit(train_feat_curr)
            oof_preds[test_index] = -model.score_samples(test_feat_curr)

    # same columns as the objectives' csvs so meta_train_features can read either
    oof_df = pd.DataFrame({f'y-train-{type}': y_train, f'{type}_oof_preds': oof_preds})

    os.makedirs(oof_path / type, exist_ok=True)
    oof_df.to_csv(oof_path / type / 'oof_trial_FINAL.csv', index=False)

    return oof_df