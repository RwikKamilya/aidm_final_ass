import time
import numpy as np
from scipy.sparse import csc_matrix

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
DATA_PATH = "data/user_movie_rating.npy"
SUB_NUM_USERS = 5_000        # subset size for experiments
JACCARD_THRESHOLD = 0.5
MAX_BUCKET_SIZE = 1000        # for LSH buckets
BASE_SEED = 42                # for reproducible experiments

# LSH configurations to test
LSH_CONFIGS = [
    {"name": "h=80,  b=16, r=5",  "h": 80,  "b": 16, "r": 5},
    {"name": "h=100, b=20, r=5",  "h": 100, "b": 20, "r": 5},
    {"name": "h=120, b=20, r=6",  "h": 120, "b": 20, "r": 6},
]


# --------------------------------------------------------
# BASIC UTILITIES: build matrix, minhash, etc.
# --------------------------------------------------------
def build_sparse_characteristic_matrix(data):
    """
    Build Movie x User CSC matrix from the full dataset.
    """
    user_ids = data[:, 0].astype(np.int64) - 1
    movie_ids = data[:, 1].astype(np.int64) - 1

    num_users = int(user_ids.max()) + 1
    num_movies = int(movie_ids.max()) + 1

    values = np.ones_like(user_ids, dtype=np.uint8)

    sparse_matrix = csc_matrix(
        (values, (movie_ids, user_ids)),
        shape=(num_movies, num_users),
        dtype=np.uint8,
    )

    # Ensure strictly 0/1
    sparse_matrix.data[:] = 1

    return sparse_matrix, num_movies, num_users


def compute_minhash_signatures_csc(mat_csc, num_hashes, rng):
    """
    Minhash using random permutations of rows, as in your main code.

    mat_csc: Movie x User CSC matrix (on subset!)
    returns signatures: shape (num_hashes, num_users)
    """
    num_movies, num_users = mat_csc.shape
    signatures = np.full(
        (num_hashes, num_users),
        fill_value=num_movies,
        dtype=np.int32,
    )

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


# --------------------------------------------------------
# GROUND TRUTH: brute-force JS > 0.5 on subset
# --------------------------------------------------------
def compute_rows_per_user(mat_csc):
    """
    Precompute the movies (rows) for each user (column) as arrays.
    """
    indptr = mat_csc.indptr
    indices = mat_csc.indices
    num_users = mat_csc.shape[1]
    rows_per_user = [indices[indptr[u]:indptr[u+1]] for u in range(num_users)]
    return rows_per_user


def brute_force_ground_truth(mat_csc, jaccard_threshold):
    """
    Compute ground-truth pairs (u,v) with JS > threshold on ALL pairs
    of users in mat_csc (Movie x User), using some pruning heuristics.

    Returns:
        ground_truth_pairs : set of (u,v) with u<v
    """
    t0 = time.time()
    num_movies, num_users = mat_csc.shape
    rows_per_user = compute_rows_per_user(mat_csc)
    lens = np.array([rows.size for rows in rows_per_user], dtype=np.int32)

    ground_truth = set()
    total_pairs = num_users * (num_users - 1) // 2
    print(f"\n[GT] Brute-force on {num_users} users: {total_pairs} pairs")

    pairs_checked = 0
    pairs_after_len_filter = 0

    for u in range(num_users - 1):
        rows_u = rows_per_user[u]
        len_u = lens[u]

        for v in range(u + 1, num_users):
            len_v = lens[v]
            pairs_checked += 1

            # Quick length-based upper bound:
            # max Jaccard <= min(len_u, len_v) / max(len_u, len_v)
            min_len = len_u if len_u < len_v else len_v
            max_len = len_u if len_u > len_v else len_v
            if min_len <= 0.5 * max_len:
                # Can't reach JS > 0.5
                continue

            pairs_after_len_filter += 1

            a = rows_u
            b = rows_per_user[v]
            ia = ib = 0
            inter = 0
            len_a = len_u
            len_b = len_v

            # Intersection must exceed (len_u + len_v)/3 for JS>0.5
            needed_inter = (len_u + len_v) / 3.0

            # Two-pointer intersection with early stopping
            while ia < len_a and ib < len_b:
                if a[ia] == b[ib]:
                    inter += 1
                    ia += 1
                    ib += 1
                elif a[ia] < b[ib]:
                    ia += 1
                else:
                    ib += 1

                # upper bound on possible intersection if all remaining matched
                remaining_possible = min(len_a - ia, len_b - ib)
                if inter + remaining_possible <= needed_inter:
                    # can't reach the required intersection anymore
                    break

            if inter == 0:
                continue

            union_size = len_u + len_v - inter
            js = inter / union_size
            if js > jaccard_threshold:
                ground_truth.add((u, v))

        # Optional small progress
        if (u + 1) % 500 == 0:
            print(f"[GT] Processed u = {u+1}/{num_users} users...")

    t1 = time.time()
    print(f"[GT] Done. Total true pairs (JS>{jaccard_threshold}): {len(ground_truth)}")
    print(f"[GT] Pairs checked: {pairs_checked}, after len filter: {pairs_after_len_filter}")
    print(f"[GT] Time for brute-force ground truth: {t1 - t0:.2f} seconds")

    return ground_truth, rows_per_user


# --------------------------------------------------------
# LSH candidate generation (no Jaccard verification here)
# --------------------------------------------------------
def lsh_generate_candidates(signatures, bands, rows_per_band, max_bucket_size):
    """
    LSH banding on the subset. Returns:
        candidate_pairs : set of (u,v) with u<v that share at least one band
        elapsed         : time spent
    """
    t0 = time.time()
    num_hashes, num_users = signatures.shape
    assert bands * rows_per_band <= num_hashes, \
        "bands * rows_per_band must be <= num_hashes"

    candidate_pairs = set()
    skipped_large_buckets = 0

    for b in range(bands):
        band_start = b * rows_per_band
        band_end = band_start + rows_per_band
        band_slice = signatures[band_start:band_end, :]  # shape (rows_per_band, num_users)

        buckets = {}
        for user in range(num_users):
            band_vec = tuple(band_slice[:, user])
            key = (b, band_vec)
            if key in buckets:
                buckets[key].append(user)
            else:
                buckets[key] = [user]

        # Generate candidate pairs
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
                    candidate_pairs.add(pair)

    t1 = time.time()
    print(f"[LSH] Candidate generation: {len(candidate_pairs)} pairs, "
          f"skipped buckets: {skipped_large_buckets}, time: {t1 - t0:.2f} s")
    return candidate_pairs, (t1 - t0)


# --------------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------------
def main_experiment():
    print(f"Loading data from {DATA_PATH} ...")
    data = np.load(DATA_PATH)
    print(f"Data shape: {data.shape}")

    print("Building full Movie x User CSC matrix...")
    char_matrix, num_movies, num_users = build_sparse_characteristic_matrix(data)
    print(f"Sparse matrix shape: {char_matrix.shape}, nnz: {char_matrix.nnz}")
    print(f"Total users: {num_users}, total movies: {num_movies}")

    # --- Sample subset of users ---
    rng = np.random.default_rng(BASE_SEED)
    if SUB_NUM_USERS > num_users:
        raise ValueError("SUB_NUM_USERS larger than total number of users")

    sub_user_indices = rng.choice(num_users, size=SUB_NUM_USERS, replace=False)
    sub_user_indices.sort()

    print(f"\nSampling {SUB_NUM_USERS} users for experiments...")
    char_sub = char_matrix[:, sub_user_indices]  # Movie x SubUsers CSC
    _, sub_num_users = char_sub.shape
    print(f"Subset matrix shape: {char_sub.shape}")

    # --- Compute ground truth on the subset ---
    ground_truth_pairs, rows_per_user_sub = brute_force_ground_truth(
        char_sub,
        JACCARD_THRESHOLD
    )
    gt_count = len(ground_truth_pairs)
    if gt_count == 0:
        print("WARNING: No ground truth pairs found with JS>0.5 on this subset.")

    # --- Run LSH configurations ---
    print("\n=== LSH CONFIGURATION EXPERIMENTS ===")
    print(f"Ground truth pair count: {gt_count}\n")

    for cfg in LSH_CONFIGS:
        h = cfg["h"]
        b = cfg["b"]
        r = cfg["r"]
        name = cfg["name"]

        if b * r > h:
            print(f"Skipping config {name}: b*r={b*r} > h={h}")
            continue

        print(f"\n--- Config: {name} ---")
        cfg_start = time.time()

        # Use fresh RNG with same base seed to isolate effect of (h,b,r),
        # not different random permutations across configs.
        rng_cfg = np.random.default_rng(BASE_SEED)

        print(f"[{name}] Computing MinHash signatures (h={h}) ...")
        sig_start = time.time()
        signatures = compute_minhash_signatures_csc(char_sub, h, rng_cfg)
        sig_end = time.time()
        print(f"[{name}] Signatures shape: {signatures.shape}, time: {sig_end - sig_start:.2f} s")

        print(f"[{name}] Running LSH banding (b={b}, r={r}) ...")
        candidates, lsh_time = lsh_generate_candidates(
            signatures,
            bands=b,
            rows_per_band=r,
            max_bucket_size=MAX_BUCKET_SIZE,
        )

        # Evaluate vs ground truth
        true_covered = len(ground_truth_pairs & candidates)
        recall = true_covered / gt_count if gt_count > 0 else 0.0
        oversampling = len(candidates) / gt_count if gt_count > 0 else float('inf')

        cfg_end = time.time()
        total_cfg_time = cfg_end - cfg_start

        print(f"[{name}] RESULTS:")
        print(f"  Ground truth pairs   : {gt_count}")
        print(f"  Candidate pairs      : {len(candidates)}")
        print(f"  True pairs covered   : {true_covered}")
        print(f"  Recall               : {recall:.4f}")
        print(f"  Candidate/GT factor  : {oversampling:.1f}x")
        print(f"  Time (signatures)    : {sig_end - sig_start:.2f} s")
        print(f"  Time (LSH banding)   : {lsh_time:.2f} s")
        print(f"  Total cfg time       : {total_cfg_time:.2f} s")

    print("\n=== Experiment complete ===")


if __name__ == "__main__":
    main_experiment()
