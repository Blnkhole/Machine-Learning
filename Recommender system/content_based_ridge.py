import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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

# optional: keep features as-is or normalize rows (cosine). For regression it's OK to leave as binary.
# You can also scale columns if needed.
# Here we don't normalize rows because ridge will learn appropriate weights.
movie_index = {mid: i for i, mid in enumerate(movies["MovieID"])}
num_users = users["UserID"].nunique()
num_features = X.shape[1]

# Hyperparams
alpha = 1.0          # Ridge regularization strength (lambda)
min_ratings = 5      # min ratings to fit per-user model

# Prepare empty user matrix
U_ridge = np.zeros((num_users, num_features))
# Optionally we can keep an intercept for each user
user_intercepts = np.zeros(num_users) #user bias

# Preprocess: build for quick lookup ratings by user
grouped = ratings.groupby("UserID")
global_mean = ratings["Rating"].mean()

for uid, group in grouped:
    uid_idx = uid - 1  # assuming UserID starts at 1 and sequential
    # get movie indices that exist in movies
    movie_ids = group["MovieID"].values
    indices = [movie_index[mid] for mid in movie_ids if mid in movie_index]
    if len(indices) == 0:
        # no rated movies that appear in movies.dat -> fallback to global mean
        user_intercepts[uid_idx] = global_mean
        continue

    X_w = X[indices] # (n_rated, n_features)
    r = group["Rating"].values[:len(indices)].astype(float)  # (film.rating)

    if len(r) < min_ratings:
        # fallback: weighted average profile (like your original code)
        # or simply use mean rating as intercept and zero weights
        w_fallback = (r @ X_w) / (r.sum()) if r.sum() != 0 else np.zeros(num_features)
        U_ridge[uid_idx] = w_fallback
        user_intercepts[uid_idx] = r.mean() if len(r) > 0 else global_mean
        continue

    # center ratings by user's mean (learn weights for deviations), keep intercept
    r_mean = r.mean() 
    r_centered = r - r_mean

    model = Ridge(alpha=alpha, fit_intercept = False)  # we handle intercept via r_mean
    model.fit(X_w, r_centered)  # learn w so that X_w @ w ~= r_centered

    w = model.coef_  # length num_features
    U_ridge[uid_idx] = w
    user_intercepts[uid_idx] = r_mean

# Predict ratings: r_pred = X @ w + intercept
R_pred = U_ridge.dot(X.T) + user_intercepts[:, None]  # shape (num_users, num_movies)

# optionally clip to rating scale 1-5
R_pred_clipped = np.clip(R_pred, 1.0, 5.0)
R_pred_clipped = np.round(R_pred_clipped, 2)

# Save predictions into DataFrame (index = UserID, columns = Title)
df = pd.DataFrame(R_pred_clipped, index = users["UserID"], columns = [f"{i}" for i in movies["Title"]])
df.to_csv("predict_movies_ridge_r.csv", encoding = "utf-8")

