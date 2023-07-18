"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Load the pickled data
y_true = pickle.load(open('y_true.p', 'rb'))

# XGB SECTION
y_pred_xgb = pickle.load(open('y_pred_xgb-p', 'rb'))

plt.figure(figsize=(8, 6))
plt.xlim(3.5, 9.5)
plt.ylim(3.5, 9.5)
plt.scatter(y_pred_xgb, y_true, alpha=0.7)
plt.xlabel('Predicted Values (y_pred_xgb)')
plt.ylabel('True Values (y_true)')
plt.title('XGBoost Model: Training Predictions vs. Actual')
plt.legend()
plt.grid(True)
plt.show()

# SVM SECTION
y_pred_svm = pickle.load(open('y_pred_svm.p', 'rb'))

plt.figure(figsize=(8, 6))
plt.xlim(3.5, 9.5)
plt.ylim(3.5, 9.5)
plt.scatter(y_pred_svm, y_true, alpha=0.7)
plt.xlabel('Predicted Values (y_pred_svm)')
plt.ylabel('True Values (y_true)')
plt.title('SVR Model: Training Predictions vs. Actual')
plt.legend()
plt.grid(True)
plt.show()

# RF SECTION
y_pred_rf = pickle.load(open('y_pred_rf.p', 'rb'))

plt.figure(figsize=(8, 6))
plt.xlim(3.5, 9.5)
plt.ylim(3.5, 9.5)
plt.scatter(y_pred_rf, y_true, alpha=0.7)
plt.xlabel('Predicted Values (y_pred_rf)')
plt.ylabel('True Values (y_true)')
plt.title('Random Forest Model: Training Predictions vs. Actual')
plt.legend()
plt.grid(True)
plt.show()

# DNN SECTION
y_pred_dnn = pickle.load(open('y_pred.p', 'rb'))

plt.figure(figsize=(8, 6))
plt.xlim(3.5, 9.5)
plt.ylim(3.5, 9.5)
plt.scatter(y_pred_dnn, y_true, alpha=0.7)
plt.xlabel('Predicted Values (y_pred_dnn)')
plt.ylabel('True Values (y_true)')
plt.title('Deep Neural Network: Training Predictions vs. Actual')
plt.legend()
plt.grid(True)
plt.show()
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import math

# Load the pickled data
y_true = pickle.load(open('y_true.p', 'rb'))

def plot_with_regression(y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.xlim(3.5, 9.5)
    plt.ylim(3.5, 9.5)
    plt.scatter(y_pred, y_true, alpha=0.7, label='Data points')
    plt.plot([3.5, 9.5], [3.5, 9.5], linestyle='--', color='gray', label='y = x')  # Straight line y = x

    # Calculate the linear regression
    reg = LinearRegression()
    reg.fit(y_pred.reshape(-1, 1), y_true)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    plt.plot([3.5, 9.5], [3.5 * slope + intercept, 9.5 * slope + intercept],
             color='orange', label=f'Regression Line ({model_name})')

    # Calculate the angle of deviation
    angle_rad = math.atan(abs(1 - slope) / (1 + slope))
    angle_deg = math.degrees(angle_rad)
    plt.text(7, 4, f'Angle of Deviation: {angle_deg:.2f} degrees', fontsize=12)

    plt.xlabel(f'Predicted Values (y_pred_{model_name})')
    plt.ylabel('True Values (y_true)')
    plt.title(f'{model_name} Model: Training Predictions vs. Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

# XGB SECTION
y_pred_xgb = pickle.load(open('y_pred_xgb-p', 'rb'))
plot_with_regression(y_pred_xgb, 'XGB')

# SVM SECTION
y_pred_svm = pickle.load(open('y_pred_svm.p', 'rb'))
plot_with_regression(y_pred_svm, 'SVM')

# RF SECTION
y_pred_rf = pickle.load(open('y_pred_rf.p', 'rb'))
plot_with_regression(y_pred_rf, 'Random Forest')

# DNN SECTION
y_pred_dnn = pickle.load(open('y_pred.p', 'rb'))
plot_with_regression(y_pred_dnn, 'Deep Neural Network')

