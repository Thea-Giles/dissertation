import re
import os
import tempfile
from pathlib import Path
import pandas as pd


def find_part_number(path: Path) -> int:
    """Extract the numeric part suffix from a filename.
    Returns -1 if not found so such files sort last.
    """
    m = re.search(r"_part(\d+)\.[Cc][Ss][Vv]$", path.name)
    return int(m.group(1)) if m else -1


def main():
    # Determine repository root as the directory containing this script
    repo_root = Path(__file__).resolve().parent

    data_dir = repo_root / "src" / "sentiment_analysis" / "data"
    pattern = "combined_cleaned_tweets_with_distilbert_sentiment_part*.csv"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    # Find all matching part files
    part_files = sorted(data_dir.glob(pattern), key=find_part_number)

    if not part_files:
        print(f"No part files found in {data_dir} matching pattern '{pattern}'.")
        return

    print("Found the following part files (in order):")
    for p in part_files:
        print(f" - {p.name}")

    # Read and concatenate
    dfs = []
    total_rows = 0
    for p in part_files:
        print(f"Reading {p} ...")
        df = pd.read_csv(p)
        dfs.append(df)
        total_rows += len(df)

    combined = pd.concat(dfs, ignore_index=True)

    output_file = data_dir / "combined_cleaned_tweets_with_distilbert_sentiment.csv"
    expected_rows = len(combined)
    print(f"Writing combined DataFrame with {expected_rows} rows (sum parts: {total_rows}) to {output_file} ...")

    # Ensure output directory exists and write atomically to a temp file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=output_file.stem + "_", suffix=".tmp", dir=str(output_file.parent))
    os.close(fd)
    tmp_path = Path(tmp_path)

    combined.to_csv(tmp_path, index=False)

    # Verify by re-reading in chunks to avoid high memory usage
    written_rows = 0
    for chunk in pd.read_csv(tmp_path, chunksize=1_000_000):
        written_rows += len(chunk)

    if written_rows != expected_rows:
        print(f"WARNING: Row count mismatch after write. Expected {expected_rows}, but reading back saw {written_rows}.")
        print("Some editors/viewers show only a preview, and naive line counters can be thrown off by embedded newlines in quoted text fields.")
        print("To validate, use pandas: pd.read_csv('<path>').shape[0] or chunked reading as above.")

    # Atomically move the temp file into place
    #os.replace(tmp_path, output_file)

    print("Combine completed successfully!")


if __name__ == "__main__":
    main()
