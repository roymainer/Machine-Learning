"""
Simple Linear Regression (SLR):
Y`i = mX + b
Trying to predict the value of future Y based on a given X, following existing results
It is related to (or equivalent to) minimizing the mean squared error (MSE)

Multiple Linear Regression model (MLR):
Y’i = b0 + b1X1i + b2X2i
"""

from sklearn import linear_model
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


X = df  # use the whole DataFrame as input
y = target["MEDV"]  # housing value

# Note the difference in argument order
lm = linear_model.LinearRegression()
model = lm.fit(X, y)  # find a linear model

predictions = lm.predict(X)  # make the predictions by the model
print(predictions[0:5])
print(lm.score(X, y))
print(lm.coef_)
print(lm.intercept_)