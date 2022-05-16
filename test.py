from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sample_data.tsv", sep='\t')

X = data.iloc[:, [2]].values.reshape(-1, 1)  # values converts it into a numpy array
Y = data.iloc[:, [3,4]].values.reshape(-1, 2)  # -1 means that calculate the dimension of rows, but have 1 column
final_linear_regressor = LinearRegression()  # create object for the class
final_linear_regressor.fit(X, Y)  # perform linear regression
final_Y_pred = final_linear_regressor.predict(X)  # make predictions
print(final_Y_pred)

print(pd.Series([v for v in final_Y_pred]))