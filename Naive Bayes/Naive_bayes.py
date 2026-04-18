from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class naive_bayes:
    
    def __init__(self, data1, data2):
        self.X = np.array(data1, dtype = "str")
        self.Y = np.array(data2, dtype = "int")  
        self.count = [defaultdict(int), defaultdict(int)] 
        self.total_words = [0, 0]  
        self.cnt = [0, 0] 
        self.vocab = set()

    def train(self):

        for i in range(len(self.X)):

            _id = self.Y[i]
            self.cnt[_id] += 1

            words = self.X[i].split()

            for w in words:
                self.count[_id][w] += 1
                self.total_words[_id] += 1
                self.vocab.add(w)

    def test(self, data1, data2, alpha) -> int:

        X_test = np.array(data1, dtype = "str")
        Y_test = np.array(data2, dtype = "int")

        p_c0 = np.log(self.cnt[0] / len(self.X))
        p_c1 = np.log(self.cnt[1] / len(self.X))

        V = len(self.vocab) 
        correct_ans = 0

        for i in range(len(X_test)):

            words = X_test[i].split()
            prob0, prob1 = p_c0, p_c1

            for w in words:
                prob0 += np.log(self.count[0][w] + alpha) - np.log(self.total_words[0] + alpha * V)
                prob1 += np.log(self.count[1][w] + alpha) - np.log(self.total_words[1] + alpha * V)

            label = 0 if prob0 > prob1 else 1
            if label == Y_test[i]: correct_ans += 1

        return correct_ans
	
# 0 : not spam; 1: spam

df_train = pd.read_csv("train.csv")
df_train["text"] = df_train["subject"] + df_train["message"]
X_train = df_train["text"]
Y_train = df_train["label"]

df_test = pd.read_csv("test.csv")
df_test["text"] = df_test["subject"] + df_test["message"]
X_test = df_test["text"]
Y_test = df_test["label"]

new_train = naive_bayes(X_train, Y_train)
new_train.train()

cross_validation = [0.01, 0.1, 0.5, 1.0, 2.0]
accuracies = list()

for i in cross_validation:
	correct_ans = new_train.test(X_test, Y_test, i)
	accuracies.append(correct_ans)

for i in range(len(accuracies)):
	accuracy = accuracies[i] / len(X_test) * 100
	print(f"Accuracy by Alpha = {cross_validation[i]} is {accuracy: .2f}%  ({accuracies[i]} / {len(X_test)})")
	accuracies[i] = accuracy

plt.plot(cross_validation, accuracies, marker='o')
plt.title("Accuracy by Alpha")
plt.xlabel("Alpha")
plt.ylabel("Accuracy (%)")
plt.xticks(cross_validation)
plt.grid(True)
plt.show()

