import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class Kmeans:

    def __init__(self, data, cluster: int, max_iter: int = 300):
        self.X = np.array(data)
        self.k = cluster
        self.max_iter = max_iter
        self.centroid_id = np.random.choice(len(self.X), self.k, replace = False)
        self.centroids = self.X[self.centroid_id]

    def get_distance(self, data1, data2) -> float:
        return np.sum((data1 - data2) ** 2) 
    
    def process(self):

        prev_loss, cur_loss = -1, 0
        labels = np.zeros(len(self.X))

        for _ in range(self.max_iter):
            for i, point in enumerate(self.X):
                distances = [self.get_distance(point, c) for c in self.centroids]
                labels[i] = np.argmin(distances)

            new_centroids = []
            for j in range(self.k):
                cluster_points = self.X[labels == j]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis = 0))
                else: 
                    new_centroids.append(self.X[np.random.randint(0, len(self.X))])
            new_centroids = np.array(new_centroids)

            prev_loss = cur_loss
            cur_loss = 0
            for i, point in enumerate(self.X):
                cur_loss += self.get_distance(point, new_centroids[int(labels[i])])

            if np.allclose(self.centroids, new_centroids) or cur_loss == prev_loss: break
            self.centroids = new_centroids
        
        self.labels = labels.astype(int)
        return self.labels, self.centroids

class Graph:

    def __init__(self, K : int):
        self.K = K
        self.colors = np.array(["red", "green", "blue", "yellow", "orange",
                   "purple", "pink", "brown", "black", "white"])

    def show(self, data, X, xlabel : str, ylabel : str, centroids = None):

        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{xlabel} and {ylabel}")

        for i in range(self.K): 
            cur = np.array(X[data == i])
            plt.scatter(cur[:, 0], cur[:, 1], label=f"Cluster{i}", color = self.colors[i], marker = "o")

        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], color = "black", marker = "x");

        plt.legend(loc = "upper right")
        plt.show()

K_cluster = 5
df = pd.read_csv("Mall_Customers.csv")
X1 = df[["Age", "Annual Income (k$)"]]
X2 = df[["Annual Income (k$)", "Spending Score (1-100)"]]

cluster1 = Kmeans(X1, K_cluster)
labels1, centroids1 = cluster1.process()

cluster2 = Kmeans(X2, K_cluster)
labels2, centroids2 = cluster2.process()

new_graph = Graph(K_cluster)
new_graph.show(labels1, X1, "Age", "Annual Income (k$)", centroids1)
new_graph.show(labels2, X2, "Annual Income (k$)", "Spending Score", centroids2)


