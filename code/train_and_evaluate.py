import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

tf.compat.v1.disable_eager_execution()

seed = 7
np.random.seed(seed)

x_df = pd.read_csv('../data/x.csv', index_col='ID')
y_df = pd.read_csv('../data/y.csv', index_col='ID')

x = x_df.values
y = y_df['pIC50'].values

x = StandardScaler().fit_transform(x)

# x_train, x_test, y_train, y_test
x, x_holdout, y, y_holdout = train_test_split(x, y, test_size=1/3)

def build_model():

    dropout_rate = 0.1
    neurons = 256
    n_layers = 16

    layer_seq = []

    while n_layers > 0:

        layer_seq.append(layers.Dense(neurons, activation='relu', kernel_initializer='he_uniform'))
        layer_seq.append(layers.Dropout(dropout_rate))

        n_layers -= 1
    
    layer_seq.append(layers.Dense(1, activation='linear'))


    model = keras.Sequential(layer_seq)

    return model

def compile_model(model):

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

monitor = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=1000,
    verbose=1,
    mode='auto',
    restore_best_weights=True,
    )

kf = KFold(n_splits=5, shuffle=True)

# train
y_true = []
y_pred = []
y_pred_svm = []
y_pred_rf = []
y_pred_xgb = []

# test
y_true_holdout = []
y_pred_holdout = []
y_pred_svm_holdout = []
y_pred_rf_holdout = []
y_pred_xgb_holdout = []

for train, test in kf.split(x):

    model = compile_model(build_model())

    history = model.fit(x[train],
                        y[train],
                        epochs=10000,
                        verbose=0,
                        validation_data=(x[test], y[test]),
                        callbacks=[monitor])

    y_pred.append(model.predict(x[test]))
    y_true.append(y[test])

    y_pred_holdout.append(model.predict(x_holdout))
    y_true_holdout.append(y_holdout)

    # added degree = 2 (changed from default 3)
    svm = SVR(C=4, degree=2, kernel='rbf', epsilon=0.12).fit(x[train], y[train])
    y_pred_svm.append(svm.predict(x[test]))

    y_pred_svm_holdout.append(svm.predict(x_holdout))

    # max_features from 0.3 to 0.2
    rf = RandomForestRegressor(n_estimators=4096, n_jobs=-1, max_features=0.2, min_samples_split=2).fit(x[train], y[train])
    y_pred_rf.append(rf.predict(x[test]))

    y_pred_rf_holdout.append(rf.predict(x_holdout))

    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror",
                              n_estimators=4096,
                              max_depth=5,
                              learning_rate=0.1,
                              colsample_bytree=0.3,
                              min_child_weight= 1,
                              subsample=0.9,
                              random_state=42)

    xgb_reg.fit(x[test], y[test])
    y_pred_xgb.append(xgb_reg.predict(x[test]))
    y_pred_xgb_holdout.append(xgb_reg.predict(x_holdout))

y_true = np.ravel(np.concatenate(y_true))
y_pred = np.ravel(np.concatenate(y_pred))
y_pred_svm = np.ravel(np.concatenate(y_pred_svm))
y_pred_rf = np.ravel(np.concatenate(y_pred_rf))

y_pred_xgb = np.ravel(np.concatenate(y_pred_xgb))

y_true_holdout = np.ravel(np.concatenate(y_true_holdout))
y_pred_holdout = np.ravel(np.concatenate(y_pred_holdout))
y_pred_svm_holdout = np.ravel(np.concatenate(y_pred_svm_holdout))
y_pred_rf_holdout = np.ravel(np.concatenate(y_pred_rf_holdout))

y_pred_xgb_holdout = np.ravel(np.concatenate(y_pred_xgb_holdout))

pickle.dump(y_true, open('y_true.p', 'wb'))
pickle.dump(y_pred, open('y_pred.p', 'wb'))
pickle.dump(y_pred_svm, open('y_pred_svm.p', 'wb'))
pickle.dump(y_pred_rf, open('y_pred_rf.p', 'wb'))

pickle.dump(y_pred_xgb, open('y_pred_xgb.p', 'wb'))

pickle.dump(y_true_holdout, open('y_true_holdout.p', 'wb'))
pickle.dump(y_pred_holdout, open('y_pred_holdout.p', 'wb'))
pickle.dump(y_pred_svm_holdout, open('y_pred_svm_holdout.p', 'wb'))
pickle.dump(y_pred_rf_holdout, open('y_pred_rf_holdout.p', 'wb'))

pickle.dump(y_pred_xgb_holdout, open('y_pred_xgb_holdout.p', 'wb'))



