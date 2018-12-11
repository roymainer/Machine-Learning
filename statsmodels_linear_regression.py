"""
Simple Linear Regression (SLR):
Y`i = mX + b
Trying to predict the value of future Y based on a given X, following existing results
It is related to (or equivalent to) minimizing the mean squared error (MSE)

Multiple Linear Regression model (MLR):
Y’i = b0 + b1X1i + b2X2i
"""

import statsmodels.api as sm
from sklearn import datasets
import numpy as np
import pandas as pd

data = datasets.load_boston()  # loads Boston housing prices datasets from datasets library
# print(data.DESCR)
# print(data.feature_names)
# print(data.target)

# define the data/predicitons as pre-set feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])


# X = df["RM"]  # average number of rooms
X = df[["RM", "LSTAT"]]  # average number of rooms and lowest statistic
y = target["MEDV"]  # housing value

# X = sm.add_constant(X)  # adds the intercept (beta_0) constant to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X)  # make the predictions by the model

# Print out the statistics
print(model.summary())
