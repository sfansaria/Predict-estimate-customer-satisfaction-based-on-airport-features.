# Name: Saba Firdaus Ansaria
# Reg No: 210110201
# you need to add the code snippet to the python widget at the Orange pipeline given
# with this assignment.
# This code takes data from the Orange pipeline and runs different regression algorithms # for generating learning curves.


# importing libraries from scikit learn using sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import learning_curve
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
import numpy as np

fet = 50
tar = 65


def print_learning_curve(reg_model, data, target, cross_validation, model):
    train_sample = [100, 200, 300, 400, 500, 1000]  # train scenarios
    # sklearn learning curve function to generate teh regression scores
    train_sample, train_scores, val_scores = learning_curve(
        estimator=reg_model,
        X=data,
        y=target, train_sizes=train_sample, cv=cross_validation, scoring='neg_mean_squared_error')
    # rpinting train and vlidation score
    print('Train score:{}'.format(train_scores))
    print('\nValidation scores:{}'.format(val_scores))

    # calculating the mean score for training and validation loss
    mean_train_scores = -train_scores.mean(axis=1)
    mean_val_scores = -val_scores.mean(axis=1)
    print('Mean training scores{}'.format(pd.Series(mean_train_scores, index=train_sample)))
    print('\n', '-' * 20)  # separator
    print('\nMean validation scores {}'.format(pd.Series(mean_val_scores, index=train_sample)))

    # plot printing

    plt.style.use('seaborn')
    # using generalised cations for multiple model training
    label = model + ' Training error'
    plt.plot(train_sample, mean_train_scores, label=label)
    label = model + ' Validation error'
    plt.plot(train_sample, mean_val_scores, label=label)
    plt.ylabel('MSE', fontsize=10)
    plt.xlabel('Training set size', fontsize=10)
    title = "Learning Curve for " + model + " model."
    plt.title(title, fontsize=12)
    plt.legend()
    plt.show()


# Name: Saba Firdaus Ansaria
# Reg No: 210110201


# parsing the data from Orange pipeline
data = in_data[:, :fet]
# parsing the target
target = in_data[:, tar]

# uncomment the portion to run linear regression
# reg_model = LinearRegression()
# print_learning_curve(reg_model,data,target,5,"Linear Regression")
# reg_model = Ridge()
# print_learning_curve(reg_model,data,target,5,"Linear Regression Ridge")
# reg_model = Lasso()
# print_learning_curve(reg_model,data,target,5,"Linear Regression Lasso")

# uncomment the portion to run Gradient boost regression
# reg_model = GradientBoostingRegressor()
# print_learning_curve(reg_model,data,target,5,"Gradient Boosting Regressor")

# uncomment the portion to run Adaboost regression
# reg_model= AdaBoostRegressor()
# print_learning_curve(reg_model,data,target,5,"Ada Boost Regressor")