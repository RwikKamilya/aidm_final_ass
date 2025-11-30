#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final assignment: Locality Sensitive Hashing for similarity search.

Usage:
    python lsh_final.py <seed> [data_path] [output_path]

Defaults:
    data_path   = "user_movie_rating.npy"
    output_path = "result.txt"

The data is expected to be an array with 3 columns:
    col 0: user_id  (int, starts from 1, contiguous)
    col 1: movie_id (int, starts from 1, contiguous)
    col 2: rating   (ignored; only presence is used)
"""

import sys
import os
from typing import List, Tuple, Iterable, Set

import numpy as np


# ------------------------ Jaccard on sorted arrays ------------------------ #

def jaccard_sorted(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two sorted 1D integer arrays (unique items).
    """
    ia = 0
    ib = 0
    na = a.size
    nb = b.size

    if na == 0 and nb == 0:
        return 0.0

    inter = 0
    union = 0

    while ia < na and ib < nb:
        va = int(a[ia])
        vb = int(b[ib])
        if va == vb:
            inter += 1
            union += 1
            ia += 1
            ib += 1
        elif va < vb:
            union += 1
            ia += 1
        else:
            union += 1
            ib += 1

    # remaining elements
    union += (na - ia) + (nb - ib)

    return inter / union if union > 0 else 0.0


# ------------------------ Data loading & preprocessing ------------------------ #

def load_data(path: str) -> np.ndarray:
    """
    Load user-movie-rating data.

    Supports:
      * .npy  → np.load
      * .csv  → np.loadtxt with comma delimiter
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".csv":
        arr = np.loadtxt(path, delimiter=",", dtype=np.int64)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected a 2D array with at least 2 columns, got shape {arr.shape}")
    return arr


def build_user_movie_lists(data: np.ndarray) -> Tuple[List[np.ndarray], int]:
    """
    Build a list of per-user movie-id arrays (0-based indices, sorted and unique).

    Returns:
        user_movies: list where user_movies[u] is a sorted np.ndarray of movie indices for user u (0-based).
        num_movies: total number of distinct movies (max movie_id)
    """
    user_ids = data[:, 0].astype(np.int64)
    movie_ids = data[:, 1].astype(np.int64)

    # IDs are assumed to start from 1 and be contiguous
    max_user = int(user_ids.max())
    max_movie = int(movie_ids.max())

    # Convert to 0-based internal indices
    user_idx = user_ids - 1
    movie_idx = movie_ids - 1

    # Sort by user index so that each user's ratings are contiguous
    order = np.argsort(user_idx, kind="mergesort")
    user_sorted = user_idx[order]
    movie_sorted = movie_idx[order]

    # We assume user ids 0..max_user-1 all appear; if not, some users will have no ratings.
    boundaries = np.empty(max_user + 1, dtype=np.int64)
    boundaries[0] = 0
    current_user = 0
    pos = 0
    n = user_sorted.size

    # Walk through sorted user indices and record start of each user block
    for u in range(max_user):
        # Move pos to first index with user_sorted[pos] == u
        while pos < n and user_sorted[pos] < u:
            pos += 1
        boundaries[u] = pos
        # Move pos to first index with user_sorted[pos] > u
        while pos < n and user_sorted[pos] == u:
            pos += 1
    boundaries[max_user] = n

    user_movies: List[np.ndarray] = []
    for u in range(max_user):
        start = int(boundaries[u])
        end = int(boundaries[u + 1])
        if start >= end:
            # user with no ratings
            user_movies.append(np.empty(0, dtype=np.int32))
        else:
            # unique + sorted movies for this user
            movies_u = np.unique(movie_sorted[start:end])
            user_movies.append(movies_u.astype(np.int32))

    return user_movies, max_movie


# ------------------------ Minhash signature computation ------------------------ #

def generate_movie_hashes(num_movies: int,
                          num_hashes: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Precompute hash values for each movie index for each hash function.

    Returns:
        movie_hashes: np.ndarray of shape (num_hashes, num_movies)
            movie_hashes[h, m] is the hash value of movie m under hash function h.
    """
    # Large prime > max(num_movies, 1)
    # 2_147_483_647 is a common choice (2^31 - 1, prime).
    prime = 2_147_483_647

    # Random coefficients for hash functions: h(x) = (a * x + b) mod prime
    a = rng.integers(1, prime, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, prime, size=num_hashes, dtype=np.int64)

    movie_indices = np.arange(num_movies, dtype=np.int64)

    # Broadcasting to compute all hash values in one go
    # shape: (num_hashes, num_movies)
    hashes = (a[:, None] * movie_indices[None, :] + b[:, None]) % prime

    # Using uint32 is usually enough and saves memory
    return hashes.astype(np.uint32)


def compute_minhash_signatures(user_movies: List[np.ndarray],
                               movie_hashes: np.ndarray) -> np.ndarray:
    """
    Compute minhash signatures for each user.

    Args:
        user_movies: list of np.ndarray (sorted movie indices per user).
        movie_hashes: np.ndarray shape (num_hashes, num_movies).

    Returns:
        signatures: np.ndarray shape (num_hashes, num_users)
    """
    num_hashes, num_movies = movie_hashes.shape
    num_users = len(user_movies)

    # Initialize with max uint32, so min() will always replace it on first real value.
    max_uint32 = np.iinfo(np.uint32).max
    signatures = np.full((num_hashes, num_users), max_uint32, dtype=np.uint32)

    for u, movies in enumerate(user_movies):
        if movies.size == 0:
            # leave the signature as all max_uint32
            continue
        # movie_hashes[:, movies] has shape (num_hashes, |movies|)
        # Take min over axis 1 to get a vector of size (num_hashes,)
        sig_u = movie_hashes[:, movies].min(axis=1)
        signatures[:, u] = sig_u

        if (u + 1) % 50_000 == 0:
            print(f"  computed signatures for {u + 1} users...", flush=True)

    return signatures


# ------------------------ LSH banding and candidate generation ------------------------ #

def lsh_and_output_pairs(user_movies: List[np.ndarray],
                         signatures: np.ndarray,
                         jaccard_threshold: float,
                         bands: int,
                         rows_per_band: int,
                         out_path: str,
                         csv_output_path: str = None,
                         max_bucket_size: int = 1000) -> None:
    """
    Apply LSH banding to the signatures, generate candidate user pairs,
    compute exact Jaccard, and write all pairs with J > threshold to out_path.

    Args:
        user_movies: list of per-user sorted movie arrays
        signatures: np.ndarray (num_hashes, num_users)
        jaccard_threshold: Jaccard similarity threshold (e.g. 0.5)
        bands: number of LSH bands
        rows_per_band: rows per band
        out_path: output file path
        csv_output_path: CSV file path
        max_bucket_size: skip buckets larger than this (to avoid quadratic explosion)
    """
    num_hashes, num_users = signatures.shape

    if bands * rows_per_band > num_hashes:
        raise ValueError(
            f"bands * rows_per_band = {bands * rows_per_band} > num_hashes = {num_hashes}"
        )

    checked_pairs: Set[Tuple[int, int]] = set()
    num_written = 0

    # Open CSV file for Jaccard similarity values if path provided
    csv_file = None
    if csv_output_path:
        csv_file = open(csv_output_path, "w")
        csv_file.write("user1,user2,jaccard_similarity\n")

    with open(out_path, "w") as out_file:
        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_sig = signatures[start:end, :]  # shape (rows_per_band, num_users)

            buckets = {}
            # Build buckets for this band
            for u in range(num_users):
                # Key: tuple of the rows_per_band hash values for this user in this band
                key = tuple(band_sig[:, u].tolist())
                # Using dict of lists; collisions => candidate pairs
                buckets.setdefault(key, []).append(u)

            # For each bucket, generate candidate pairs
            for bucket_users in buckets.values():
                k = len(bucket_users)
                if k < 2 or k > max_bucket_size:
                    # Ignore trivial or overly large buckets
                    continue

                # Generate all unique user pairs in this bucket
                for i in range(k):
                    ui = bucket_users[i]
                    for j in range(i + 1, k):
                        uj = bucket_users[j]
                        if ui == uj:
                            continue
                        # Enforce (small, large) ordering for deduplication
                        if ui < uj:
                            pair = (ui, uj)
                        else:
                            pair = (uj, ui)

                        if pair in checked_pairs:
                            continue
                        checked_pairs.add(pair)

                        # Compute exact Jaccard on the original sets
                        js = jaccard_sorted(user_movies[pair[0]], user_movies[pair[1]])
                        if js > jaccard_threshold:
                            # Convert back to original 1-based user IDs
                            u1_out = pair[0] + 1
                            u2_out = pair[1] + 1
                            out_file.write(f"{u1_out},{u2_out}\n")
                            
                            # write to csv
                            if csv_file:
                                csv_file.write(f"{u1_out},{u2_out},{js:.6f}\n")
                            
                            num_written += 1

            print(f"Finished band {b + 1}/{bands}, pairs written so far: {num_written}", flush=True)

    # Close CSV file if it was opened
    if csv_file:
        csv_file.close()
        

    print(f"Done. Total similar pairs written to {out_path}: {num_written}")


# ------------------------ Main ------------------------ #

def main(argv: Iterable[str]) -> None:
    # if len(argv) < 2:
    #     print("Usage: python lsh_final.py <seed> [data_path] [output_path]")
    #     print("  data_path   default: user_movie_rating.npy")
    #     print("  output_path default: result.txt")
    #     sys.exit(1)

    seed = int(argv[1]) if len(argv) >= 2 else 42
    data_path = argv[2] if len(argv) >= 3 else "user_movie_rating.npy"
    out_path = argv[3] if len(argv) >= 4 else "result.txt"
    csv_out_path = "jaccard_similarities.csv"

    print(f"Random seed: {seed}")
    print(f"Data path:   {data_path}")
    print(f"Output path: {out_path}")
    print(f"CSV output:  {csv_out_path}")

    rng = np.random.default_rng(seed)

    # Hyperparameters – you can tune these (also discuss in the report)
    num_hashes = 100        # signature length
    bands = 20              # number of bands
    rows_per_band = 5       # rows per band (bands * rows_per_band <= num_hashes)
    jaccard_threshold = 0.5
    max_bucket_size = 1000  # skip very large buckets

    print("Loading data...")
    data = load_data(data_path)
    print(f"Data shape: {data.shape}")

    print("Building per-user movie lists...")
    user_movies, num_movies = build_user_movie_lists(data)
    num_users = len(user_movies)

    print(f"Number of users:  {num_users}")
    print(f"Number of movies: {num_movies}")

    print(f"Generating {num_hashes} hash functions over {num_movies} movies...")
    movie_hashes = generate_movie_hashes(num_movies=num_movies,
                                         num_hashes=num_hashes,
                                         rng=rng)

    print("Computing minhash signatures...")
    signatures = compute_minhash_signatures(user_movies=user_movies,
                                            movie_hashes=movie_hashes)
    print(f"Signature matrix shape: {signatures.shape}")

    print("Running LSH and writing similar user pairs...")
    lsh_and_output_pairs(user_movies=user_movies,
                         signatures=signatures,
                         jaccard_threshold=jaccard_threshold,
                         bands=bands,
                         rows_per_band=rows_per_band,
                         out_path=out_path,
                         csv_output_path=csv_out_path,
                         max_bucket_size=max_bucket_size)


if __name__ == "__main__":
    main(sys.argv)
