from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

iso_table = pd.read_csv('results/iso_table.csv')
xgb_table = pd.read_csv('results/xgb_table.csv')
meta_table = pd.read_csv('results/log_reg.csv')

# get params for best runs


# load params into models


# train on full training dataset



# final test
