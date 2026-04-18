import pandas as pd
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv("creditcard.csv")

train_df, test_df = train_test_split(df, test_size = 0.25, random_state = 42, stratify = df['Class']) #stratify: ensure class ratio

# Optional: Save to CSV
train_df.to_csv('creditcard_train.csv', index=False)
test_df.to_csv('creditcard_test.csv', index=False)


#plt.xticks(k_values): labels axis with value of k_value
