from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DecisionTreeNode:

    def __init__(self, data, features, depth = 0):
        self.data = data                 
        self.children = {}                
        self.depth = depth
        self.features = features    
        self.next = None  
        self.label = None              
        self.split_attr = None                  

class decision_tree:

    def __init__(self, data, max_height = None):
        self.X = data
        self.root = None
        self.max_height = max_height

    def get_entropy(self, labels):

        total = len(labels)
        count = Counter(labels)

        return -sum((c / total) * np.log(c / total) for c in count.values())

    def information_gain(self, data, feature):

        base_entropy = self.get_entropy(data["class"])
        values = data[feature].unique()
        weighted_entropy = 0

        for v in values:
            subset = data[data[feature] == v]
            weight = len(subset) / len(data)
            weighted_entropy += weight * self.get_entropy(subset["class"])

        return base_entropy - weighted_entropy

    def train(self, features):

        self.root = DecisionTreeNode(self.X, features)
        current = self.root

        while current:

            data = current.data
            labels = data["class"]

            if (self.get_entropy(labels) == 0 or len(current.features) == 0
               or (self.max_height is not None and current.depth >= self.max_height)):
                current.label = labels.mode()[0]
                current = current.next
                continue

            max_ig, best_attr = -1, None

            for feat in current.features:
                ig = self.information_gain(data, feat)
                if ig > max_ig: [max_ig, best_attr] = [ig, feat]

            if best_attr is None:
                current.label = labels.mode()[0]
                current = current.next
                continue

            current.split_attr = best_attr
            new_features = [i for i in current.features if i != best_attr]

            for val in data[best_attr].unique():

                subset = data[data[best_attr] == val]

                child = DecisionTreeNode(subset, new_features, current.depth + 1)
                current.children[val] = child

                child.next = current.next
                current.next = child

            current = current.next

    def predict(self, x) -> str:

        node = self.root

        while node.label is None:
            attr = node.split_attr
            val = x.get(attr)

            if val not in node.children: return "NULL"
            node = node.children[val]

        return node.label

    def Test(self, dataset, correct_labels) -> int:

        correct_ans = 0

        for i in range(len(dataset)): 
            predict_labels = self.predict(dataset.iloc[i])  #dataset.iloc[i] : (dict){key_i: value_i}
            correct_ans += 1 if predict_labels == correct_labels[i] else 0

        return correct_ans

df_train = pd.read_csv("train.csv")
features = [col for col in df_train.columns if col != "class"]
X_train = df_train

df_test = pd.read_csv("test.csv")
X_test = df_test.drop(columns = "class")
Y_test = df_test["class"]
size = len(Y_test)

max_height = [5, 6, 7, 8, 9, 10, 11, 15] #cross validation
correct_ans = [0] * len(max_height)
accuracies = [0] * len(max_height)

for i in range(len(max_height)):
    tree = decision_tree(X_train, i)
    tree.train(features)
    correct_ans[i] = tree.Test(X_test, Y_test)

for i in range(len(max_height)):
    accuracies[i] = correct_ans[i] / size * 100
    print(f"Accuracy by maximum height = {max_height[i]} is {accuracies[i] : .3f}%   ({correct_ans[i]} / {size})")

plt.plot(max_height, accuracies, marker="o")
plt.title("Accuracies by maximum height")
plt.xlabel("Maximum height")
plt.ylabel("Accuracy (%)")
plt.xticks(max_height)
plt.grid(True)
plt.show()
