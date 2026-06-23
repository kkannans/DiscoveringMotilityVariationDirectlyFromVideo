"""
Creates data_split_{seed}.json.
1) Get batches in range 68 to 349 from dataset_folder
2) Sort batches by video duration from low to high
3) Split into 80% train, 10% val, 10% test per batch such that we sample randomly from sorted batches.
"""
import hashlib
import json
from pathlib import Path
import argparse
import numpy as np
import os
import random

def get_batch_number(batch_id):
    """Extract numeric part from batch name (e.g., 'batch-000115' -> 115)."""
    stem = batch_id.stem if isinstance(batch_id, Path) else Path(batch_id).stem
    if stem.startswith('batch-'):
        return int(stem.replace('batch-', ''))
    else:
        return int(stem)

def _print_distribution_stats(data_split_dict: dict, duration_info: dict):
    """Print video duration distribution statistics for each split."""
    print("\n" + "=" * 60)
    print("Duration Distribution Statistics")
    print("=" * 60)

    for split_name, batches in data_split_dict.items():
        durations = [duration_info[b]["duration_seconds"] for b in batches]
        print(f"\n{split_name.upper()} ({len(batches)} batches):")
        print(f"  Min:    {np.min(durations):.2f}s")
        print(f"  Max:    {np.max(durations):.2f}s")
        print(f"  Mean:   {np.mean(durations):.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")
        print(f"  Std:    {np.std(durations):.2f}s")


def create_seed_split(
    dataset_folder: Path,
    duration_info: dict,
    seed: int,
) -> dict:
    """
    Create data split with STRATIFIED sampling to ensure similar
    duration distributions across train/val/test.
    """
    # Get batch names in range 68-349
    batch_names = [
        batch.name.replace(".mp4", "")
        for batch in dataset_folder.glob("*.mp4")
        if batch.is_file() and 68 <= get_batch_number(batch) <= 349
    ]
    print(f"Found {len(batch_names)} batches in range 68-349")

    # Sort by duration
    sorted_batch_ids = sorted(
        batch_names, key=lambda x: duration_info[x]["duration_seconds"]
    )

    rng = np.random.RandomState(seed)

    # === STRATIFIED APPROACH ===
    # Divide sorted batches into strata (bins), then sample proportionally from each

    n_strata = 10  # Number of bins across duration range
    strata_size = len(sorted_batch_ids) // n_strata

    train_batches = []
    val_batches = []
    test_batches = []

    for i in range(n_strata):
        start_idx = i * strata_size
        # Last stratum gets remainder
        end_idx = (
            (i + 1) * strata_size if i < n_strata - 1 else len(sorted_batch_ids)
        )
        stratum = sorted_batch_ids[start_idx:end_idx]

        # Shuffle within stratum (local RNG for reproducibility)
        stratum = np.array(stratum)
        rng.shuffle(stratum)

        # Split 80/10/10 within each stratum; guarantee at least 1 val and 1 test when possible
        n = len(stratum)
        if n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 1
        else:
            n_train = max(1, int(n * 0.8))
            n_val = max(1, int(n * 0.1))
            if n_train + n_val >= n:
                n_val = 1
                n_train = n - 2  # leave 1 for test

        train_batches.extend(stratum[:n_train].tolist())
        val_batches.extend(stratum[n_train : n_train + n_val].tolist())
        test_batches.extend(stratum[n_train + n_val :].tolist())

    # Verify no overlap
    assert len(set(train_batches) & set(val_batches)) == 0
    assert len(set(train_batches) & set(test_batches)) == 0
    assert len(set(val_batches) & set(test_batches)) == 0

    total = len(train_batches) + len(val_batches) + len(test_batches)
    assert total == len(sorted_batch_ids), f"Lost batches: {total} vs {len(sorted_batch_ids)}"

    data_split_dict = {
        "train": train_batches,
        "val": val_batches,
        "test": test_batches,
    }

    # Print distribution stats for verification
    _print_distribution_stats(data_split_dict, duration_info)

    return data_split_dict

def create_covering_splits(
    dataset_folder: Path,
    duration_info: dict,
    n_seeds: int = 10,
    master_seed: int = 42,
) -> dict:
    """
    Generate n_seeds splits where every organoid appears in >=1 test set.

    Algorithm:
    1. Sort organoids by video duration, divide into strata
    2. Within each stratum, round-robin assign organoids to seed test sets
    3. Fill remaining train/val slots randomly per seed
    4. Guarantees full test coverage + stratified distributions

    Returns: {seed_idx: {"train": [...], "val": [...], "test": [...]}}
             seed_idx is 1-indexed (1..n_seeds)
    """
    # Get batch names in range 68-349
    batch_names = [
        batch.name.replace(".mp4", "")
        for batch in dataset_folder.glob("*.mp4")
        if batch.is_file() and 68 <= get_batch_number(batch) <= 349
    ]
    print(f"Found {len(batch_names)} batches in range 68-349")

    # Sort by duration
    sorted_batch_ids = sorted(
        batch_names, key=lambda x: duration_info[x]["duration_seconds"]
    )
    N = len(sorted_batch_ids)

    rng = np.random.RandomState(master_seed)

    # Divide into strata
    n_strata = 10
    strata_size = N // n_strata
    strata = []
    for i in range(n_strata):
        start = i * strata_size
        end = (i + 1) * strata_size if i < n_strata - 1 else N
        stratum = sorted_batch_ids[start:end]
        rng.shuffle(stratum)  # shuffle within stratum
        strata.append(list(stratum))

    # For each stratum, determine how many test slots per seed
    # Target: ~10% of N per seed for test => ~N*0.1/n_strata per stratum per seed
    # But we need to cover all organoids, so use round-robin

    # Initialize per-seed test/val/train lists
    all_splits = {s: {"test": [], "val": [], "train": []} for s in range(1, n_seeds + 1)}

    for stratum in strata:
        S = len(stratum)
        # How many test slots per seed from this stratum?
        # Total test per seed across all strata should be ~N*0.1 = ~11
        # Per stratum: ~S*0.1 = ~1.1, so 1 per stratum per seed (with some getting 2)
        test_per_seed = max(1, round(S * 0.1))

        # Round-robin: assign each organoid to a seed's test set
        # This guarantees every organoid is tested at least once
        assignments = {}  # organoid -> list of seeds where it's in test
        for b in stratum:
            assignments[b] = []

        # Phase 1: Ensure coverage — each organoid gets assigned to at least one test set
        organoid_list = list(stratum)
        rng.shuffle(organoid_list)
        seed_cycle = list(range(1, n_seeds + 1))

        # Track how many test slots each seed has used from this stratum
        seed_test_count = {s: 0 for s in range(1, n_seeds + 1)}

        # Assign each organoid to a seed's test set (round-robin)
        for idx, b in enumerate(organoid_list):
            # Pick seed with fewest test assignments so far (greedy balancing)
            candidate_seeds = sorted(seed_cycle, key=lambda s: seed_test_count[s])
            chosen_seed = candidate_seeds[0]
            assignments[b].append(chosen_seed)
            seed_test_count[chosen_seed] += 1

        # Phase 2: Fill remaining test slots to reach test_per_seed per seed
        for s in range(1, n_seeds + 1):
            deficit = test_per_seed - seed_test_count[s]
            if deficit > 0:
                # Pick organoids not already in this seed's test
                already_test = {b for b, seeds_list in assignments.items() if s in seeds_list}
                candidates = [b for b in stratum if b not in already_test]
                rng.shuffle(candidates)
                for b in candidates[:deficit]:
                    assignments[b].append(s)
                    seed_test_count[s] += 1

        # Now assign organoids to splits for each seed
        for s in range(1, n_seeds + 1):
            test_orgs = [b for b in stratum if s in assignments[b]]
            remaining = [b for b in stratum if s not in assignments[b]]
            rng.shuffle(remaining)

            # Val: ~10% of stratum from remaining
            n_val = max(1, round(len(remaining) * 0.11))  # slightly >10% of remaining to hit ~10% of total
            if n_val > len(remaining):
                n_val = max(0, len(remaining))

            val_orgs = remaining[:n_val]
            train_orgs = remaining[n_val:]

            all_splits[s]["test"].extend(test_orgs)
            all_splits[s]["val"].extend(val_orgs)
            all_splits[s]["train"].extend(train_orgs)

    # Verify
    for s in range(1, n_seeds + 1):
        sp = all_splits[s]
        t_set, v_set, te_set = set(sp["train"]), set(sp["val"]), set(sp["test"])
        assert len(t_set & v_set) == 0, f"Seed {s}: train/val overlap"
        assert len(t_set & te_set) == 0, f"Seed {s}: train/test overlap"
        assert len(v_set & te_set) == 0, f"Seed {s}: val/test overlap"
        total = len(sp["train"]) + len(sp["val"]) + len(sp["test"])
        assert total == N, f"Seed {s}: lost batches {total} vs {N}"

    # Verify full coverage
    all_tested = set()
    for s in range(1, n_seeds + 1):
        all_tested.update(all_splits[s]["test"])
    assert all_tested == set(sorted_batch_ids), (
        f"Coverage gap: {len(all_tested)}/{N} organoids covered. "
        f"Missing: {set(sorted_batch_ids) - all_tested}"
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Covering splits generated ({n_seeds} seeds, {N} organoids)")
    print(f"{'='*60}")
    for s in range(1, n_seeds + 1):
        sp = all_splits[s]
        print(f"  Seed {s:2d}: train={len(sp['train']):3d}, val={len(sp['val']):3d}, test={len(sp['test']):3d}")
    print(f"  Total test coverage: {len(all_tested)}/{N}")

    # Print per-seed distribution stats
    for s in range(1, n_seeds + 1):
        print(f"\n--- Seed {s} ---")
        _print_distribution_stats(all_splits[s], duration_info)

    return all_splits


def ensure_data_split_exists(path_config, seed: int) -> Path:
    """Create data_split_{seed}.json if it doesn't already exist. Returns the split file path."""
    project_root = Path(__file__).resolve().parent
    output_dir = (project_root / path_config.get_datainfo_path()).resolve()
    split_file = output_dir / f"data_split_{seed}.json"

    if split_file.exists():
        return split_file

    dataset_folder = (project_root / path_config.data_path).resolve()
    assert dataset_folder.exists(), f"Dataset folder {dataset_folder} does not exist"

    duration_info_path = dataset_folder.parent.parent / "intervention_info.json"
    assert duration_info_path.exists(), f"Duration info not found: {duration_info_path}"
    with open(duration_info_path, "r") as f:
        duration_info = json.load(f)

    print(f"Data split not found at {split_file}. Creating...", flush=True)
    data_split_dict = create_seed_split(
        dataset_folder=dataset_folder,
        duration_info=duration_info,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(data_split_dict, f, indent=2)
    print(f"Data split saved to: {split_file}", flush=True)
    return split_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create data splits (single seed or covering set)"
    )
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed (single-seed mode)")
    parser.add_argument("--covering", action="store_true",
                        help="Generate covering splits that guarantee full test coverage")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds for covering mode (default: 10)")
    parser.add_argument("--master-seed", type=int, default=42,
                        help="Master seed for covering mode (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: config datainfo path)")
    args = parser.parse_args()

    from config import PathConfig

    path_config = PathConfig()
    project_root = Path(__file__).resolve().parent
    dataset_folder = (project_root / path_config.data_path).resolve()

    assert dataset_folder.exists(), f"Dataset folder {dataset_folder} does not exist"

    # duration_info.json (historically named intervention_info.json) lives above the video directory
    duration_info_path = dataset_folder.parent.parent / "intervention_info.json"
    assert duration_info_path.exists(), f"Duration info not found: {duration_info_path}"
    with open(duration_info_path, 'r') as f:
        duration_info = json.load(f)

    if args.covering:
        # Generate covering splits
        print(f"\n{'='*60}")
        print(f"Creating COVERING data splits:")
        print(f"  N seeds: {args.n_seeds}")
        print(f"  Master seed: {args.master_seed}")
        print(f"{'='*60}\n")

        all_splits = create_covering_splits(
            dataset_folder=dataset_folder,
            duration_info=duration_info,
            n_seeds=args.n_seeds,
            master_seed=args.master_seed,
        )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = (project_root / path_config.get_datainfo_path()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for seed_idx, split_dict in all_splits.items():
            output_file = output_dir / f"data_split_covering_{seed_idx}.json"
            with open(output_file, 'w') as f:
                json.dump(split_dict, f, indent=2)
            print(f"Saved: {output_file}")

        print(f"\n{'='*60}")
        print(f"All {args.n_seeds} covering splits saved to: {output_dir}")
        print(f"{'='*60}\n")
    else:
        # Original single-seed mode
        print(f"\n{'='*60}")
        print(f"Creating data split with:")
        print(f"  Seed: {args.seed}")
        print(f"{'='*60}\n")

        data_split_dict = create_seed_split(
            dataset_folder=dataset_folder,
            duration_info=duration_info,
            seed=args.seed,
        )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = (project_root / path_config.get_datainfo_path()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"data_split_{args.seed}.json"

        with open(output_file, 'w') as f:
            json.dump(data_split_dict, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Data split saved to: {output_file}")
        print(f"{'='*60}\n")
