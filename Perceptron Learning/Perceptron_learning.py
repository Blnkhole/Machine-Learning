import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class perceptron:

	def __init__(self, data, label):
		self.X = np.array(data, dtype = float) 
		self.labels = np.unique(label)              
		label_map = {self.labels[0]: -1, self.labels[1]: +1}
		self.Y = np.array([label_map[i] for i in label], dtype = int)

	def _sign(self, cur_W, x) -> int:
		product = np.dot(cur_W, x)
		return 1 if product >= 0 else -1

	def train(self, max_loops = 100):

		ones = np.ones((len(self.X), 1))
		self.X = np.concatenate((ones, self.X), axis = 1)
		W = np.zeros(self.X.shape[1])  
		loss = -1

		for loop in range(max_loops):
			cur_loss = 0
			for i in range(len(self.X)):
				y_true = self.Y[i]           
				y_pred = self._sign(W, self.X[i])
				if y_pred == 0: y_pred = 1 
				if y_true != y_pred: 
					cur_loss += 1
					W = W + y_true * self.X[i]  
			loss = cur_loss
			if loss == 0:  break
		return W

	def test(self, W, dataset, label) -> int:
		ones = np.ones((len(dataset), 1))
		dataset = np.concatenate((ones, dataset), axis = 1)
		correct_ans = 0
		for i in range(len(dataset)):
			pred = np.sign(np.dot(W, dataset[i])) 
			pred = 0 if pred == -1 else 1
			if label[i] == self.labels[pred]: correct_ans += 1
		return correct_ans

df_train = pd.read_csv("train.csv")
X_train = df_train.drop(columns = "species")
Y_train = df_train["species"]

df_test = pd.read_csv("test.csv")
X_test = df_test.drop(columns = "species")
Y_test = df_test["species"]

perc = perceptron(X_train, Y_train)
W = perc.train()
correct_ans = perc.test(W, X_test, Y_test)
print(f"Accuracy is {correct_ans / len(Y_test) * 100 :.3f}%")

