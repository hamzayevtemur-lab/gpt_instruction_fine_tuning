import hashlib
import json
import os
import random


def hash_sample(item):
    """Creates a unique deterministic fingerprint for each prompt."""
    key = f"{item.get('instruction', '').strip()} ||| {item.get('input', '').strip()}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def create_reproducible_splits(
    source_path="data/alpaca_data.json",
    train_size=9500,
    val_size=500,
    test_size=500,
    seed=42,
    output_dir="data",
):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source dataset not found at: {source_path}")

    print(f"Loading raw data from {source_path} ...")
    with open(source_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Total raw examples: {len(raw_data)}")

    # 1. Deduplicate by instruction + input
    seen_hashes = set()
    unique_data = []
    for item in raw_data:
        h = hash_sample(item)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_data.append(item)

    print(f"Unique examples after deduplication: {len(unique_data)}")

    # 2. Shuffle with fixed random seed
    random.seed(seed)
    random.shuffle(unique_data)

    total_needed = train_size + val_size + test_size
    if len(unique_data) < total_needed:
        raise ValueError(
            f"Not enough unique examples ({len(unique_data)}) for required {total_needed}"
        )

    # 3. Create non-overlapping slices
    train_data = unique_data[:train_size]
    val_data = unique_data[train_size : train_size + val_size]
    test_data = unique_data[
        train_size + val_size : train_size + val_size + test_size
    ]

    # 4. Rigorous verification: Verify 0% overlap
    train_hashes = set(hash_sample(x) for x in train_data)
    val_hashes = set(hash_sample(x) for x in val_data)
    test_hashes = set(hash_sample(x) for x in test_data)

    assert len(train_hashes.intersection(val_hashes)) == 0, (
        "Train and Val sets overlap!"
    )
    assert len(train_hashes.intersection(test_hashes)) == 0, (
        "Train and Test sets overlap!"
    )
    assert len(val_hashes.intersection(test_hashes)) == 0, (
        "Val and Test sets overlap!"
    )

    # 5. Save splits to disk
    splits = {"train": train_data, "val": val_data, "test": test_data}

    for name, data in splits.items():
        out_file = os.path.join(output_dir, f"{name}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {name.upper()}: {len(data)} samples -> {out_file}")

    print("\n✓ Split creation successful! 0% overlap verified across all splits.")


if __name__ == "__main__":
    create_reproducible_splits()
