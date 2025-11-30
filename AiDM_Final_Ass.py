import time
import numpy as np
from scipy.sparse import csc_matrix

seed = 99
rng = np.random.default_rng(seed)


start_time = time.time()
print("Loading dataset...")
data = np.load('user_movie_rating.npy') # change this 
print(f"Data shape: {data.shape}")


def build_sparse_characteristic_matrix(data):
    # IDs start at 1, convert to 0-based
    user_ids = data[:, 0].astype(np.int64) - 1
    movie_ids = data[:, 1].astype(np.int64) - 1

    num_users = int(user_ids.max()) + 1
    num_movies = int(movie_ids.max()) + 1

    values = np.ones_like(user_ids, dtype=np.uint8)

    sparse_matrix = csc_matrix(
        (values, (movie_ids, user_ids)),
        shape=(num_movies, num_users),
        dtype=np.uint8
    )

    return sparse_matrix, num_movies, num_users

print("Building sparse Movie x User matrix (CSC)...")
char_matrix, num_movies, num_users = build_sparse_characteristic_matrix(data)
print(f"Sparse matrix shape: {char_matrix.shape}, nnz={char_matrix.nnz}")


def compute_minhash_signatures_vectorized(mat_csc, num_hashes, rng):
    
    num_movies, num_users = mat_csc.shape
    signatures = np.full((num_hashes, num_users), num_movies, dtype=np.int32)

    mat_csr = mat_csc.tocsr()  

    for h in range(num_hashes):
        # Random permutation of rows (movies)
        perm = rng.permutation(num_movies)
        permuted_mat = mat_csr[perm, :]

        # Convert to CSC 
        permuted_csc = permuted_mat.tocsc()
        indptr = permuted_csc.indptr
        indices = permuted_csc.indices

        # Initialize signature row with max value
        sig_row = np.full(num_users, num_movies, dtype=np.int32)

        # Vectorized min over each column
        for col in range(num_users):
            start = indptr[col]
            end = indptr[col + 1]
            if start < end:
                sig_row[col] = indices[start:end].min()

        signatures[h, :] = sig_row

    return signatures
    
num_hashes = 100 
print(f"Computing MinHash signatures with {num_hashes} hashes...")
signatures = compute_minhash_signatures_vectorized(char_matrix, num_hashes, rng)
print(f"Signature matrix shape: {signatures.shape}")
print(signatures)

end_time = time.time()
print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
