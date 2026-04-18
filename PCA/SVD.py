import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("creditcard.csv")
X = data.drop(columns = {"Time", "Class"})

X_norm = StandardScaler().fit_transform(X) #chuan hoa X, // do Amount gay mat can bang 
U, S, VT = svd(X_norm, full_matrices = False)

cur = [5, 10, 15, 20, 25, 29]
frobineous = []

for k in cur:
	X_reduced = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
	loss = np.linalg.norm(X_norm - X_reduced, "fro") / np.linalg.norm(X_norm, "fro")
	frobineous.append(float(loss))

print(frobineous)

plt.scatter(cur, frobineous)
plt.xlabel("K")
plt.ylabel("Fro_norm")
plt.plot(cur, frobineous, color = "black", label = "Loss")
plt.show()
