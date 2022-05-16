import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class regressor:
    def __init__(self, data:pd.DataFrame, cardinal_dims:list, result_dims:list):
        self.data = data
        self.cardinal_dims = cardinal_dims
        self.result_dims = result_dims
        self.X = []
        self.Y = []
        self.Y_pred = []
        self.regression = None
        self.generate_regression()
        return

    def get_Y_pred_Series(self) -> pd.Series:
        return pd.Series([v for v in self.Y_pred])

    def generate_regression(self) -> None:
        self.X = self.data.iloc[:, self.cardinal_dims].values.reshape(-1, len(self.cardinal_dims))  # values converts it into a numpy array
        self.Y = self.data.iloc[:, self.result_dims].values.reshape(-1, len(self.result_dims))  # -1 means that calculate the dimension of rows
        self.linear_regressor = LinearRegression()  # create object for the class
        self.linear_regressor.fit(self.X, self.Y)  # perform linear regression
        self.Y_pred = self.linear_regressor.predict(self.X)  # make predictions
        return

    def print_regression_string(self) -> None:
        print("Regression: Result ~ {}*Cardinal_1 + {}*Cardinal_2 {} {}".format(self.linear_regressor.coef_[0,0],
                self.linear_regressor.coef_[0,1],
                ("-" if self.linear_regressor.intercept_[0] < 0.0 else "+"),
                abs(self.linear_regressor.intercept_[0])))
        return

    def plot_regression(self, axis1:int = 0, axis2:int = 1) -> None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.X[:,axis1], self.X[:,axis2], self.Y, c=self.Y_pred[:,0], cmap='Greens')
        ax.set_xlabel('Cardinal 1')
        ax.set_ylabel('Cardinal 2')
        ax.set_zlabel('Result')
        plt.show()
        return