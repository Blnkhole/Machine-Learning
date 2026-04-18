import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np
import pandas as pd

class logistics_regression:

	def __init__(self, data, label):

		self.X = data 
		self.Y = np.array(label, dtype = int)
		self.n_samples, self.n_features = self.X.shape

	def sigmoid(self, x):
		return expit(x)

	def train(self, lr = 0.01, max_loops = 1000):

		W = np.zeros(self.n_features)

		for loop in range(max_loops):
			for i in range(self.n_samples):
				z = np.dot(self.X[i], W) 
				p_x = self.sigmoid(z)
				W -= lr * (p_x - self.Y[i]) * self.X[i] #SGD cap nhat truc tiep 
		return W

	def test(self, W, data, correct_label) -> int:

		correct_label = np.array(correct_label)

		X = np.array(data, dtype = float)
		correct_ans = 0

		for i in range(len(X)):
			probs = self.sigmoid(np.dot(X[i], W))
			pre = int(probs >= 0.5)
			if pre == correct_label[i]: correct_ans += 1

		return correct_ans


df_train = pd.read_csv("train.csv")
df_train["Gender"] = df_train["Gender"].map({"Male": 0, "Female": 1})

X_train = np.array(df_train[["Gender", "Age", "EstimatedSalary"]], dtype = float)
Y_train = df_train["Purchased"]

mean = np.mean(X_train, axis = 0)
std = np.std(X_train, axis = 0)
X_train = (X_train - mean) / std

df_test = pd.read_csv("test.csv")
df_test["Gender"] = df_test["Gender"].map({"Male": 0, "Female": 1})
X_test = np.array(df_test[["Gender", "Age", "EstimatedSalary"]])
Y_test = df_test["Purchased"]

mean = np.mean(X_test, axis = 0)
std = np.std(X_test, axis = 0)
X_test = (X_test - mean) / std

learning_rate = [0.001, 0.01, 0.1, 0.5, 1.0]

regression = logistics_regression(X_train, Y_train)
W = regression.train(0.01)
ans = regression.test(W, X_test, Y_test)

for i in learning_rate:
	W = regression.train(i)
	ans = regression.test(W, X_test, Y_test)
	print(f"Accuracy : {ans / len(X_test) * 100 : .3}%")

