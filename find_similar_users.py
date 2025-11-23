import time

import numpy as np
from scipy.sparse import csc_matrix
import os

start_time = time.time()

seed = 42
data_path = "data/user_movie_rating.npy"
out_path = "result.txt"

print(f"Random seed: {seed}")
print(f"Data path:   {data_path}")
print(f"Output path: {out_path}")

rng = np.random.default_rng(seed)

num_hashes = 100
bands = 20
rows_per_band = 5
jaccard_threshold = 0.5
max_bucket_size = 1000


def build_sparse_characteristic_matrix(data):
    user_ids = data[:, 0].astype(np.int64) - 1
    movie_ids = data[:, 1].astype(np.int64) - 1

    num_users = int(user_ids.max()) + 1
    num_movies = int(movie_ids.max()) + 1

    values = np.ones_like(user_ids, dtype=np.uint8)

    sparse_matrix = csc_matrix((values, (movie_ids, user_ids)),
                         shape=(num_movies, num_users),
                         dtype=np.uint8)

    sparse_matrix.data[:] = 1

    return sparse_matrix, num_movies, num_users

def compute_minhash_signatures_csc(mat_csc, num_hashes, rng):
    num_movies, num_users = mat_csc.shape
    signatures = np.full((num_hashes, num_users), fill_value=num_movies,
                         dtype=np.int32)

    indptr = mat_csc.indptr
    indices = mat_csc.indices

    for h in range(num_hashes):
        perm = rng.permutation(num_movies)
        rank = np.empty(num_movies, dtype=np.int32)
        rank[perm] = np.arange(num_movies, dtype=np.int32)

        sig_h = signatures[h]
        for j in range(num_users):
            start = indptr[j]
            end = indptr[j + 1]
            if start == end:
                continue
            rows = indices[start:end]
            sig_h[j] = rank[rows].min()

    return signatures

print("Loading data...")
data = np.load(data_path)
print(f"Data shape: {data.shape}")

print("Building sparse Movie x User matrix (CSC)...")
char_matrix, num_movies, num_users = build_sparse_characteristic_matrix(data)
print(f"Sparse matrix shape: {char_matrix.shape}")
print(f"  nnz (non-zeros):   {char_matrix.nnz}")

print(f"Number of users:  {num_users}")
print(f"Number of movies: {num_movies}")


print(f"Computing MinHash signatures with {num_hashes} permutations...")
signatures = compute_minhash_signatures_csc(char_matrix, num_hashes, rng)
print(f"Signature matrix shape: {signatures.shape}")

end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")

