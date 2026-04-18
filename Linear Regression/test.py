import pandas as pd

# Load your data
df = pd.read_csv("housing.csv")

# pd.get_dummies() is a Pandas function that automatically converts categorical (text) data into numeric binary columns (0 or 1) — also called one-hot encoding.
view_encoded = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity')

# Add the new columns to the original dataframe
df_with_view = pd.concat([df, view_encoded], axis=1)

# Save it as a new CSV (optional)
df_with_view.to_csv("housing_with_view_onehot.csv", index=False)

