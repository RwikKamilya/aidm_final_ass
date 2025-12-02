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

def compute_jaccard_from_csc(char_matrix, u, v):
    indptr = char_matrix.indptr
    indices = char_matrix.indices

    # Movies rated by user u
    start_u, end_u = indptr[u], indptr[u + 1]
    rows_u = indices[start_u:end_u]

    # Movies rated by user v
    start_v, end_v = indptr[v], indptr[v + 1]
    rows_v = indices[start_v:end_v]

    if rows_u.size == 0 and rows_v.size == 0:
        return 0.0

    inter = np.intersect1d(rows_u, rows_v, assume_unique=True)
    inter_size = inter.size

    union_size = rows_u.size + rows_v.size - inter_size
    if union_size == 0:
        return 0.0

    return inter_size / union_size


def lsh_find_similar_users(signatures,
                           char_matrix,
                           bands,
                           rows_per_band,
                           jaccard_threshold,
                           max_bucket_size,
                           out_path):
    print("\n=== Starting LSH banding and candidate generation ===")

    num_hashes, num_users = signatures.shape
    assert bands * rows_per_band <= num_hashes, \
        "bands * rows_per_band must be <= num_hashes"

    lsh_start = time.time()

    candidate_pairs = set()
    total_buckets = 0
    skipped_large_buckets = 0

    for b in range(bands):
        band_start = b * rows_per_band
        band_end = band_start + rows_per_band
        print(f"\n-- Band {b + 1}/{bands}: rows [{band_start}:{band_end}) --")

        band_slice = signatures[band_start:band_end, :]

        buckets = {}
        for user in range(num_users):
            band_vec = tuple(band_slice[:, user])
            key = (b, band_vec)

            if key in buckets:
                buckets[key].append(user)
            else:
                buckets[key] = [user]

        num_buckets = len(buckets)
        total_buckets += num_buckets
        print(f"Band {b + 1}: formed {num_buckets} buckets")

        band_candidates = 0
        for key, users in buckets.items():
            k = len(users)
            if k < 2:
                continue
            if k > max_bucket_size:
                skipped_large_buckets += 1
                continue

            for i in range(k):
                ui = users[i]
                for j in range(i + 1, k):
                    uj = users[j]
                    if ui < uj:
                        pair = (ui, uj)
                    else:
                        pair = (uj, ui)
                    if pair not in candidate_pairs:
                        candidate_pairs.add(pair)
                        band_candidates += 1

        print(f"Band {b + 1}: added {band_candidates} new candidate pairs "
              f"(skipped {skipped_large_buckets} large buckets so far)")

    print("\n=== LSH candidate generation complete ===")
    print(f"Total unique candidate pairs: {len(candidate_pairs)}")
    print(f"Total buckets across all bands: {total_buckets}")
    print(f"Total skipped large buckets (> {max_bucket_size}): {skipped_large_buckets}")

    print("\n=== Verifying candidate pairs with exact Jaccard ===")
    verify_start = time.time()

    num_similar_pairs = 0
    with open(out_path, "w") as f_out:
        for idx, (u, v) in enumerate(candidate_pairs):
            if idx % 10000 == 0 and idx > 0:
                print(f"  Verified {idx}/{len(candidate_pairs)} candidate pairs...")

            js = compute_jaccard_from_csc(char_matrix, u, v)
            if js >= jaccard_threshold:
                num_similar_pairs += 1
                # Convert back to 1-based user IDs when writing
                f_out.write(f"{u + 1},{v + 1}\n")

    verify_end = time.time()

    print("\n=== Verification complete ===")
    print(f"Number of pairs with Jaccard >= {jaccard_threshold}: {num_similar_pairs}")
    print(f"Time spent in LSH banding   : {verify_start - lsh_start:.2f} seconds")
    print(f"Time spent in verification  : {verify_end - verify_start:.2f} seconds")
    print(f"Time spent in LSH + verify  : {verify_end - lsh_start:.2f} seconds")


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

print("Running LSH to find similar user pairs...")
lsh_find_similar_users(
    signatures=signatures,
    char_matrix=char_matrix,
    bands=bands,
    rows_per_band=rows_per_band,
    jaccard_threshold=jaccard_threshold,
    max_bucket_size=max_bucket_size,
    out_path=out_path,
)

end_time = time.time()
print(f"Time elapsed: {end_time - start_time}")

