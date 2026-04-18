import pandas as pd

df1 = pd.read_csv("mnist_train.csv")
df2 = pd.read_csv("mnist_test.csv")

merged = pd.concat([df1, df2], ignore_index = True)
merged.to_csv("mnist.csv", index = False)