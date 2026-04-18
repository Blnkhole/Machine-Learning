import numpy as np
import pandas as pd

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
similarity_matrix = np.zeros((num_users, num_users))

similarity_matrix = U_normalized @ U_normalized.T

R_df = ratings.pivot(index = "UserID", columns = "MovieID", values = "Rating").fillna(0)
R = R_df.values
num_users, num_movies = R.shape

def predict_rating_user_based(u, i, similarity_matrix, R, k = 5):

    sims = similarity_matrix[u]
    rated_users = np.where(R[:, i] > 0)[0]

    if len(rated_users) == 0: return 0

    sims_users = [(v, sims[v]) for v in rated_users if v != u]
    sims_users.sort(key = lambda x: x[1], reverse = True)
    top_k = sims_users[:k]

    if len(top_k) == 0: return 0

    num = sum(sim * R[v, i] for v, sim in top_k)
    den = sum(abs(sim) for _, sim in top_k)

    return num / den if den != 0 else 0

R_pred = np.zeros_like(R)

for u in range(num_users):
    for i in range(num_movies):
        if R[u, i] == 0:  
        	R_pred[u, i] = predict_rating_user_based(u, i, similarity_matrix, R)
        else: R_pred[u, i] = R[u, i] 

R_pred_scaled = 1 + 4 * (R_pred - R_pred.min()) / (R_pred.max() - R_pred.min())

df = pd.DataFrame(
    R_pred_scaled,
    index = users["UserID"],
    columns = [f"{i}" for i in movies["Title"]])

df.to_csv("predict_mvs.csv", encoding = "utf-8")