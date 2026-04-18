import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

class K_Nearest_Neighbors:

    def __init__(self, data1, data2):
        
        self.X = np.array(data1)
        self.Y = np.array(data2).astype(str)  

    def distance(self, data, k: int) -> str:

        label_count = defaultdict(int)
        z = np.sum(self.X * self.X, axis=1)
        x = np.sum(data * data, axis=1)
        dist = z.reshape(-1, 1) + x.reshape(1, -1) - 2 * np.dot(self.X, data.T)

        dist = np.dstack((dist.reshape(1, -1), self.Y)).reshape(-1, 2)
        dist[:, 0] = dist[:, 0].astype(float)
        sorted_dist = dist[dist[:, 0].argsort()]

        for i in range(min(k, sorted_dist.shape[0])):
            label = sorted_dist[i][1]
            label_count[label] += 1

        return max(label_count, key = label_count.get)

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

X_train = df_train.drop(columns = "species")
Y_train = df_train["species"]

X_test = np.array(df_test.drop(columns="species"))
Y_test = np.array(df_test["species"]).astype(str)

model = K_Nearest_Neighbors(X_train, Y_train)

k_values = [_ for _ in range(1, 21)]
accuracies = []

for K in k_values:
    correct = 0
    for i, sample in enumerate(X_test):
        pred = model.distance(np.array([sample]), K)
        if pred == Y_test[i]: correct += 1
    acc = correct / len(Y_test) * 100
    accuracies.append(acc)
    print(f"K = {K}, Accuracy = {acc:.3f}%")

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker="o") 
plt.title("Accuracy by K value")
plt.xlabel("K")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.xticks(k_values)
plt.show()
