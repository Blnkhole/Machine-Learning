import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("creditcard.csv")
X = data.drop(columns = {"Time", "Class"})

X_norm = StandardScaler().fit_transform(X)

pca_model = PCA()
Z = pca_model.fit_transform(X_norm) 

threshold = 0.95
explained_variance_ratio = pca_model.explained_variance_ratio_ #get var 
cumulative_variance = np.cumsum(explained_variance_ratio) #cumsum like prefix sum
k = np.argmax(cumulative_variance >= threshold) + 1

print(k)

plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), marker = 'o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Elbow Plot")
plt.grid(True)
plt.show()
