import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv("housing_with_view_onehot.csv").drop(columns = "ocean_proximity")
for i in range(9, 14):
	df.iloc[:, [i]] = df.iloc[:, [i]].astype(int)

train_df, test_df = train_test_split(df, test_size=0.2) #random_state=42)

train_df.to_csv("train.csv", index = False)
test_df.to_csv("test.csv", index = False)


#dfa = np.array(df.iloc[:, [10, 11, 12, 13, 14]]) select columns by index number

