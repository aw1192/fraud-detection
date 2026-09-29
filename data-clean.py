import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

os.makedirs('data/splits', exist_ok = True)

with open('config.yaml') as f:
    config = yaml.safe_load(f)

SEED = config['seed']

data = pd.read_csv('data/raw/creditcard.csv')

data_plus = data.copy()

# feature engineering
data_plus['time-of-day'] = data_plus['Time'] % 86400
data_plus['hour-of-day'] = data_plus['time-of-day'] / 3600
data_plus['sin-hour'] = np.sin((2*np.pi*data_plus['hour-of-day'])/24)
data_plus['cos-hour'] = np.cos((2*np.pi*data_plus['hour-of-day'])/24)

# clean dataset and label dataset
clean  = data_plus.copy().drop(['Time', 'time-of-day', 'hour-of-day', 'Class'], axis = 1)
labels = data_plus['Class'].copy()

# data splitting into train and test
X, X_test, y, y_test = train_test_split(
    clean, labels,
    test_size = 0.15,
    stratify = labels,
    random_state = SEED
)
 
# split the base into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size = 0.15/0.85,
    stratify = y,
    random_state = SEED
)

# get a 70/15/15 split
X_train, X_val, y_train, y_val, X, y, X_test, y_test = (d.reset_index(drop=True) for d in (X_train, X_val, y_train, y_val, X, y, X_test, y_test))

X_train.to_csv('data/splits/X_train.csv', index = False)
X_val.to_csv('data/splits/X_val.csv', index = False)
y_train.to_csv('data/splits/y_train.csv', index = False)
y_val.to_csv('data/splits/y_val.csv', index = False)
X_test.to_csv('data/splits/X_test.csv', index = False)
y_test.to_csv('data/splits/y_test.csv', index = False)
X.to_csv('data/splits/X.csv', index = False)
y.to_csv('data/splits/y.csv', index = False)

print('Success, split data into training (and split training), validation, and test.')
