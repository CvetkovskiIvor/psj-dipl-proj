import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

seed = 7
np.random.seed(seed)

y_df = pd.read_csv('../../data/y.csv', index_col='ID')
y = y_df['pIC50'].values

x_df = pd.read_csv('../../data/intermediate_data/rrelieff_descriptors.csv', index_col='ID')
x = x_df.values

x = StandardScaler().fit_transform(x)

x, x_holdout, y, y_holdout = train_test_split(x, y, test_size=0.333)

n_estimators = [128, 248, 512, 1024, 2048, 4096]
max_depth = [3, 5, 7]
learning_rate = [0.1, 0.2]
min_child_weight = [1, 3, 5]
subsample = [0.6, 0.7, 0.8, 0.9]
colsample_bytree = [0.3, 0.5]

parameters = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                  min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree)

model = xgb.XGBRegressor()
grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=2)
grid_result = grid.fit(x, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
