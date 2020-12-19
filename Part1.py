# Katelin Prazdnik
# FE595 Assignment 4 - SKLearn Review Part 1
# December 18, 2020
# I pledge on my honor that I have not given or received any unauthorized assistance on
# this assignment/examination. I further pledge that I have not copied any material from
# a book, article, the Internet or any other source except where I have expressly cited the
# source.

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# Define Linear Regression variable
regression = LinearRegression()

# Upload the Boston data set from SKLearn
data = load_boston()

# Create predictors and target dataframes
predictors = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])

# Run Linear Regression
regression.fit(predictors, target)

# Create output table of coefficients in descending order
output = ()
output = (pd.DataFrame(list(data.feature_names)).copy())
output.insert(len(output.columns), "Coefficients", regression.coef_.transpose())
output = output.sort_values(by=["Coefficients"], ascending=False)

# Edit output table so that there is no index
new_output = output.to_string(index=False)
print(new_output)
