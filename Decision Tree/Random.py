import pandas as pd
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv("Car_evaluation.csv")

train_df, test_df = train_test_split(df, test_size=0.7)

# Optional: Save to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)


