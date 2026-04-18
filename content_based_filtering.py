import numpy as np
import pandas as pd
import sklearn

movies = pd.read_csv(
    "movies.dat",
    sep = "::",                
    engine = "python",        
    names = ["MovieID", "Title", "Genres"],
    encoding = "latin-1")

users = pd.read_csv(
    "users.dat",
    sep = "::",
    engine = "python", 
    names = ["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    encoding = "latin-1")

ratings = pd.read_csv(
    "ratings.dat",
    sep = "::",
    engine = "python",
    names = ["UserID", "MovieID", "Rating", "Timestamp"],
    encoding = "latin-1").drop(columns = ["Timestamp"])

genres_list = movies["Genres"].apply(lambda x: x.split("|"))

unique_genres = sorted(set(g for sublist in genres_list for g in sublist))
print(unique_genres)

X = np.zeros((len(movies), len(unique_genres)), dtype = int)

for i, genres in enumerate(genres_list):
    for g in genres:
        j = unique_genres.index(g)
        X[i, j] = 1

norms = np.linalg.norm(X, axis = 1, keepdims = True) #get norm l2
norms[norms == 0] = 1
X_normalized = X / norms

movie_index = {mid: i for i, mid in enumerate(movies["MovieID"])}
num_users = ratings["UserID"].nunique()
num_features = X_normalized.shape[1]

U = np.zeros((num_users, num_features))

for uid in ratings["UserID"].unique():

    user_ratings = ratings[ratings["UserID"] == uid]
    indices = [movie_index[mid] for mid in user_ratings["MovieID"] if mid in movie_index]
    
    if len(indices) == 0: continue
    
    movie_vecs = X_normalized[indices]
    rating_values = user_ratings["Rating"].values[:len(indices)]
    
    user_vector = np.dot(rating_values, movie_vecs) / np.sum(rating_values)
    
    U[uid - 1] = user_vector  

norms = np.linalg.norm(U, axis = 1, keepdims = True) #get norm l2
norms[norms == 0] = 1

U_normalized = U / norms

R_pred = np.dot(U_normalized, X_normalized.T)
R_pred_scaled = 1 + 4 * (R_pred - R_pred.min()) / (R_pred.max() - R_pred.min())
R_pred_scaled = np.round(R_pred_scaled, 2)

#missing_ids = set(ratings["MovieID"]) - set(movies["MovieID"])
#print(missing_ids)

df = pd.DataFrame(
    R_pred_scaled,
    index = users["UserID"],
    columns = [f"{i}" for i in movies["Title"]])

#df.to_csv("predict_movies_r.csv", encoding = "utf-8")

rating_true = ratings.pivot(index="UserID", columns="MovieID", values="Rating")
rating_true = rating_true.reindex(index=users["UserID"], columns=movies["MovieID"])

# Ghép MovieID ↔ Title cho thống nhất
pred_df = pd.DataFrame(
    R_pred,
    index=users["UserID"],
    columns=movies["MovieID"])

mask = ~rating_true.isna()
y_true = rating_true[mask]
y_pred = pred_df[mask]

# Error metrics
mse = ((y_true - y_pred) ** 2).mean().mean()
rmse = np.sqrt(mse)
mae = (y_true - y_pred).abs().mean().mean()

print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")

# ==========================
# 7️⃣ Precision@K, Recall@K, F1@K
# ==========================
def precision_recall_at_k(pred, true, k=10, threshold=4.0):
    precisions, recalls = [], []

    for user in pred.index:
        if user not in true.index:
            continue
        top_k = pred.loc[user].nlargest(k).index
        relevant = true.loc[user][true.loc[user] >= threshold].index
        if len(relevant) == 0:
            continue
        hit = len(set(top_k) & set(relevant))
        precisions.append(hit / k)
        recalls.append(hit / len(relevant))

    p = np.mean(precisions)
    r = np.mean(recalls)
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1

p, r, f1 = precision_recall_at_k(pred_df, rating_true, k=10, threshold=4.0)
print(f"Precision@10 = {p:.4f}")
print(f"Recall@10    = {r:.4f}")
print(f"F1@10        = {f1:.4f}")
