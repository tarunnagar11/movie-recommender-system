# generate_pickles.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

# Load CSVs
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Pivot table
final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating")
final_dataset.fillna(0, inplace=True)

# Filter
no_user_voted = ratings.groupby("movieId")['rating'].count()
no_movies_voted = ratings.groupby("userId")['rating'].count()

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Create CSR matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Train model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Save all required objects
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(final_dataset, open('final_dataset.pkl', 'wb'))
pickle.dump(knn, open('knn_model.pkl', 'wb'))
pickle.dump(csr_data, open('csr_data.pkl', 'wb'))
pickle.dump(movies['title'].tolist(), open('movierecommendation_list.pkl', 'wb'))

print("✅ All .pkl files have been generated.")
