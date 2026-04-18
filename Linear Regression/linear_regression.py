import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class linear_regression:
    def __init__(self, data1, data2):
        self.X = np.array(data1, dtype = float)
        self.Y = np.array(data2, dtype = float)

    def training(self) -> np.array:
        ones = np.ones((self.X.shape[0], 1))
        _X = np.concatenate((ones, self.X), axis = 1)
        W = np.dot(np.dot(np.linalg.pinv(np.dot(_X.T, _X)), _X.T), self.Y)
        return W

class Test:
    def __init__(self, data1, data2):
        self.X_test = np.array(data1, dtype = float)
        self.W = np.array(data2, dtype = float)

    def cross_test(self) -> np.array:
        return np.dot(self.X_test, self.W)

test_df = pd.read_csv("train.csv").fillna(50)
X = np.array(test_df.drop(columns = "median_house_value"), dtype = float)
Y = np.array(test_df["median_house_value"], dtype = float)

task = linear_regression(X, Y)
W = task.training()

train_df = pd.read_csv("test.csv").fillna(50)
X_Test = np.array(train_df.drop(columns= "median_house_value"), dtype = float)
Y_Result = np.array(train_df["median_house_value"], dtype = float)

ones = np.ones((X_Test.shape[0], 1))
X_Test = np.concatenate((ones, X_Test), axis = 1)

test = Test(X_Test, W)
Y_Test = test.cross_test()

board = np.vstack((Y_Test, Y_Result)).T
print(board)

#plt.scatter(X, Y, color = "purple")
#plt.xlabel("Size_m2") 
#plt.ylabel("Price_k$")
#plt.plot(X, Y_training, label = "Trained model", color = "black") 
#plt.scatter(X_Test, Y_Result, color = "red") 
#plt.xlabel = "Size_m2" #plt.ylabel = "Price_$" 
#plt.legend(loc = "lower right") 
#plt.show()

#np.nan(X_test).any() check any of nan in X_test
#df.isna().sum() count nan