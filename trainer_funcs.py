# OBJECTIVE FUNCTIONS

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
from sklearn.preprocessing import StandardScaler

with open("config.yaml") as f:
    config = yaml.safe_load(f)

SEED = config['seed']
oof_path = config['oof_path']

# XGBOOST TREE OBJECTIVE FUNCTION

def xgb_objective(trial, sgkf, X_train, y_train): 
    legit_over_fraud = ((len(y_train)-sum(y_train))/sum(y_train))
    objective = 'binary:logitraw'
    max_rounds = 3000
    early_stopping_rounds = 200

    # hyperparameters
    scale_pos_weight = trial.suggest_float('scale_pos_weight', 0.2*legit_over_fraud, 5 * legit_over_fraud, log=True) 
    eta = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0.001, 10, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1)
    min_child_weight = trial.suggest_int('min_child_weight', 5, 100)
    
    with mlflow.start_run(nested=True, run_name=f'xgb_trial_{trial.number}') as child_run:
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
        evals = {}

        oof_preds = np.zeros(len(X_train))
        aucpr_scores = []
        mlflow.set_tag("trial-number", trial.number)

        for i, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train)):
            train_feat_curr = X_train.iloc[train_index]
            train_label_curr = y_train.iloc[train_index]

            test_feat_curr = X_train.iloc[test_index]
            test_label_curr = y_train.iloc[test_index]

            dtrain= xgb.DMatrix(train_feat_curr, label=train_label_curr)
            dval = xgb.DMatrix(test_feat_curr, label = test_label_curr)
            
            bst = xgb.train(
                params = params,
                dtrain = dtrain,
                num_boost_round = max_rounds,
                evals = [(dtrain, 'dtrain'), (dval, 'dval')],
                early_stopping_rounds = early_stopping_rounds,
                evals_result = evals,
                verbose_eval = int(early_stopping_rounds/2)
            )
        
            val_scores = bst.predict(dval, iteration_range=(0, bst.best_iteration+1)) # output np array
            y_val_true = dval.get_label()
            aucpr = average_precision_score(y_val_true, val_scores)

            oof_preds[test_index] = val_scores
            aucpr_scores.append(aucpr)

        mean_aucpr = sum(aucpr_scores)/len(aucpr_scores)
        mlflow.log_metric('mean-aucpr', mean_aucpr)

        oof_df = pd.DataFrame({'y-train-xgb': y_train, 'xgb_oof_preds': oof_preds})

        csv_path = f'{oof_path}/xgb/oof_trial_{trial.number}.csv'
        oof_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        return mean_aucpr

# ISOLATION TREE OBJECTIVE FUNCTION

def iso_objective(trial, sgkf, X_train, y_train):

    with mlflow.start_run(nested=True, run_name=f'iso_trial_{trial.number}') as child_run:
        oof_preds = np.zeros(len(X_train))
        aucpr_scores = []

        mlflow.set_tag("trial-number", trial.number)

        
        for i, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train)):
            
            train_feat_curr = X_train.iloc[train_index]
            test_feat_curr = X_train.iloc[test_index]

            model = IsolationForest(random_state = SEED,
                                    n_estimators = trial.suggest_int('n_estimators', 75, 300),
                                    max_samples = trial.suggest_int('max_samples', 50, 500),
                                    max_features = trial.suggest_float('max_features', 0.1, 1)                                
            ).fit(train_feat_curr)
            
            scores = -model.score_samples(test_feat_curr)
            oof_preds[test_index] = scores

            y_val_curr = y_train.iloc[test_index]
            aucpr = average_precision_score(y_val_curr, scores)
            aucpr_scores.append(aucpr)

        mean_aucpr = sum(aucpr_scores)/len(aucpr_scores)
        mlflow.log_metric('mean-aucpr', mean_aucpr)
        
        
        oof_df = pd.DataFrame({'y-train-iso': y_train, 'iso_oof_preds': oof_preds})
        
        csv_path = f'{oof_path}/iso/oof_trial_{trial.number}.csv'
        oof_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        return mean_aucpr

# SUMMARY TABLE OF RUNS

def summary_runs(experiment_id, final):
    if final:
        runs_df = mlflow.search_runs(
        experiment_ids=[f'{experiment_id}'],
        order_by=['metrics.aucpr DESC'] # for the final logistic regression sort by aucpr
        )
    else:
        runs_df = mlflow.search_runs(
        experiment_ids=[f'{experiment_id}'],
        order_by=['metrics.`mean-aucpr` DESC'] # for channels sort by the mean aucpr
)

    return runs_df

# FULL TRAIN/TUNE

def train_tune(name, objective, n_trials, final):
    mlflow.set_experiment(f'{name}')
    

    with mlflow.start_run(run_name = 'study') as run:
        mlflow.log_param('n_trials', n_trials)
    
        study = optuna.create_study(direction = 'maximize')

        if objective == iso_objective:
            study.enqueue_trial({'n_estimators': 100, 'max_samples': 256, 'max_features': 1.0})
        elif objective == xgb_objective:
            study.enqueue_trial({
                'learning_rate': 0.3,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1.0,
            })

        else:
            study.enqueue_trial({
                'penalty': 'l2',
                'C': 1.0
            })
        
        study.optimize(objective, n_trials=n_trials)
    
        # log best trial
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_mean_aucpr": study.best_value})

    
    experiment_id = mlflow.get_experiment_by_name(f'{name}').experiment_id

    if final:
        return summary_runs(experiment_id, final=final)
    else:
        return summary_runs(experiment_id, final = final)
    
# LOAD BEST RUN

def best_model(channel: Literal['xgboost', 'sklearn'], table):
    run_id = table.iloc[0]["run_id"]
    exp_id = mlflow.get_run(run_id).info.experiment_id

    logged = mlflow.search_logged_models(
        experiment_ids=[exp_id], output_format="list"
    )
    matches = [m for m in logged if m.source_run_id == run_id]
    if not matches:
        raise ValueError(f"No logged model found for run {run_id}")

    uri = f"models:/{matches[0].model_id}"
    if channel.lower() == "xgboost":
        return mlflow.xgboost.load_model(uri)
    return mlflow.sklearn.load_model(uri)

# FIND BEST TRIAL CSV

def best_csv(model: Literal['xgb', 'iso'], table):
    csv_idx = table.iloc[0]['tags.trial-number']
    
    if model == 'xgb':
        best_csv = pd.read_csv(f'oof_scores/xgb/oof_trial_{csv_idx}.csv')
    elif model == 'iso':
        best_csv = pd.read_csv(f'oof_scores/iso/oof_trial_{csv_idx}.csv')
    else:
        raise Exception('Not valid model type.') 

    return best_csv

# GET THE METAFEATURES OF TRAINING SET
def meta_train_features(xgb_table, iso_table):
    concat = pd.concat([best_csv('xgb', xgb_table), best_csv('iso', iso_table)], axis=1)
    concat = concat.drop(columns = 'y-train-iso')
    concat.rename(columns = {'y-train-xgb':'y-train'}, inplace = True)
    
    scaler = StandardScaler()

    # removes the base 2 exponential
    log_iso = -np.log2(np.array(concat['iso_oof_preds']).reshape(-1,1)) # outputs E[h(x)]/c(n)
    concat['iso_oof_preds'] = scaler.fit_transform(log_iso)  

    concat['xgb_oof_preds'] = scaler.fit_transform(np.array(concat['xgb_oof_preds']).reshape(-1,1))

    meta_feat_data = concat[['xgb_oof_preds', 'iso_oof_preds']]
    meta_feat_labels = concat['y-train']
    
    return meta_feat_data, meta_feat_labels

# GENERATE META-VALIDATION FEATURES
def meta_val_features(X_val, y_val, xgb_best, iso_best):
    dval= xgb.DMatrix(X_val, label = y_val)
    # generating meta validation features
    meta_xgb = xgb_best.predict(dval)
    meta_iso = iso_best.score_samples(X_val)
    meta_val = pd.concat({'xgb_oof_preds': pd.Series(meta_xgb), 'iso_oof_preds': pd.Series(meta_iso)}, axis=1)
    return meta_val

# META-LEARNER OBJECTIVE

def meta_objective(trial, concat_data, concat_labels, meta_val, y_val):
    
    with mlflow.start_run(nested=True, run_name=f'meta_trial_{trial.number}') as child_run:
        model = LogisticRegression(random_state = SEED,
                               penalty = 'l2',
                               C = trial.suggest_float('C', 0.5, 10)
        ).fit(concat_data, concat_labels)
        preds = model.predict_proba(meta_val)[:,1]
        aucpr = average_precision_score(y_val, preds)
        mlflow.log_metric('aucpr', aucpr)
        
        return aucpr  

def get_best_model():
    df = pd.read_csv('results/log_reg.csv')
    run_id = df.iloc[0]['run_id']
    run = mlflow.get_run(run_id)

    model_id = run.outputs.model_outputs[0].model_id
    model = mlflow.sklearn.load_model(f"models:/{model_id}")

    return model
        

