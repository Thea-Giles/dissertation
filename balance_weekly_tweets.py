import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


def find_input_file(repo_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        p = explicit if explicit.is_absolute() else (repo_root / explicit)
        return p if p.exists() else None
    # Preferred path per issue description
    candidates: List[Path] = [
        repo_root / "data" / "combined_cleaned_tweets_with_distilbert_sentiment.csv",
        repo_root / "src" / "sentiment_analysis" / "data" / "combined_cleaned_tweets_with_distilbert_sentiment.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def try_parse_datetime(s: pd.Series) -> Tuple[pd.Series, float]:
    """Attempt to parse a Series to datetime using several strategies.
    Returns a tuple (parsed_series, valid_ratio).
    """
    if s is None or len(s) == 0:
        return pd.to_datetime(pd.Series([], dtype=object)), 0.0

    non_null = s.notna().sum()

    # Attempt 0: exact ISO8601 with timezone (e.g., 2020-04-09 23:59:51+00:00)
    parsed0 = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S%z", errors='coerce')
    valid0 = parsed0.notna().sum()
    ratio0 = (valid0 / non_null) if non_null else 0.0

    # Attempt 1: generic string parsing
    parsed1 = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
    valid1 = parsed1.notna().sum()
    ratio1 = (valid1 / non_null) if non_null else 0.0

    # Attempt 2: dayfirst for D/M/Y formats
    parsed2 = pd.to_datetime(s, errors='coerce', dayfirst=True, infer_datetime_format=True)
    valid2 = parsed2.notna().sum()
    ratio2 = (valid2 / non_null) if non_null else 0.0

    # Choose best among string-based attempts
    best_parsed = parsed0
    best_ratio = ratio0
    if ratio1 > best_ratio:
        best_parsed, best_ratio = parsed1, ratio1
    if ratio2 > best_ratio:
        best_parsed, best_ratio = parsed2, ratio2

    # Attempt 3: numeric unix epochs (s, ms, us, ns)
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().any():
        med = float(num.dropna().median())
        unit = 's'
        if med >= 1e18:
            unit = 'ns'
        elif med >= 1e15:
            unit = 'us'
        elif med >= 1e12:
            unit = 'ms'
        else:
            unit = 's'
        parsed_num = pd.to_datetime(num, unit=unit, origin='unix', errors='coerce')
        validn = parsed_num.notna().sum()
        ration = (validn / non_null) if non_null else 0.0
        if ration > best_ratio:
            best_parsed, best_ratio = parsed_num, ration

    return best_parsed, best_ratio


def robust_to_datetime(s: pd.Series) -> pd.Series:
    parsed, _ = try_parse_datetime(s)
    return parsed


def detect_datetime_column(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    # Strong preference: use 'date' column if present, per dataset specification
    if 'date' in df.columns:
        return 'date'
    # Candidate datetime column names in this project context
    candidates = [
        'created_at', 'created', 'timestamp', 'time', 'datetime'
    ]
    # Heuristic: any column containing 'date' or 'time'
    dynamic = [c for c in df.columns if ('date' in c.lower()) or ('time' in c.lower())]
    seen = []
    for name in candidates + dynamic:
        if name in df.columns and name not in seen:
            seen.append(name)
    if not seen:
        return None
    # Choose the column with the highest successful parse ratio
    best_col = None
    best_ratio = -1.0
    for col in seen:
        _, ratio = try_parse_datetime(df[col])
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col
    return best_col if best_ratio > 0 else None


def week_key(series: pd.Series, anchor: str = "MON") -> pd.Series:
    # Convert to weekly period anchored at anchor day; default Monday
    try:
        return series.dt.to_period(f"W-{anchor}")
    except Exception:
        # Fallback to default weekly
        return series.dt.to_period("W")


def balance_by_week(
    df: pd.DataFrame,
    dt_col: str,
    rng: np.random.RandomState,
    week_anchor: str = "MON",
    min_total: int = 100_000,
    max_imbalance: float = 0.20,
) -> pd.DataFrame:
    """Balance dataset by week to achieve near-even distribution within tolerance,
    while keeping at least min_total rows if feasible.

    Strategy:
    - Parse datetime robustly and compute weekly counts.
    - Let K be number of kept weeks; require each kept week to have at least
      target_min = ceil(min_total / K) tweets available, and cap per-week at
      cap = ceil(target_min * (1 + max_imbalance)).
    - Iterate K until stable; exclude sparse weeks. Sample each kept week to
      min(count, cap). Guarantees post max/min <= 1 + max_imbalance and total >= min_total
      when feasible.
    """
    df = df.copy()
    df[dt_col] = robust_to_datetime(df[dt_col])
    before_len = len(df)
    df = df.dropna(subset=[dt_col])
    if len(df) < before_len:
        print(f"Warning: Dropped {before_len - len(df)} rows with non-parsable dates from column '{dt_col}'.")

    # Compute weekly groups
    df['_week'] = week_key(df[dt_col], anchor=week_anchor)
    counts = df.groupby('_week').size().sort_index()
    if counts.empty:
        print("No weekly groups found after parsing dates. Nothing to balance.")
        return df.drop(columns=['_week'])

    total_available = int(counts.sum())
    if min_total > total_available:
        print(f"Requested min_total={min_total} exceeds available rows {total_available}. Clamping to available.")
        min_total = total_available

    max_count = int(counts.max())

    # Start with all non-zero weeks
    nonzero_counts = counts[counts > 0]
    if nonzero_counts.empty:
        print("All weekly counts are zero after parsing dates.")
        return df.drop(columns=['_week'])

    kept_mask = nonzero_counts.index  # start with all
    K = len(kept_mask)

    # Iterate to a stable set of weeks
    max_iters = 20
    for it in range(max_iters):
        if K == 0:
            break
        target_min = int(np.ceil(min_total / K))
        cap = int(min(max_count, np.ceil(target_min * (1.0 + max_imbalance))))
        # Weeks must have at least target_min available to be kept
        new_kept = nonzero_counts[nonzero_counts >= target_min].index
        K_new = len(new_kept)
        print(f"Iteration {it+1}: K={K} -> {K_new}, target_min={target_min}, cap={cap}")
        if K_new == K:
            kept_mask = new_kept
            break
        K = K_new
        kept_mask = new_kept
    else:
        print("Reached max iterations when selecting weeks; proceeding with current selection.")

    if K == 0:
        # Fallback: include all nonzero weeks, relax requirement but keep as many rows as possible
        kept_mask = nonzero_counts.index
        K = len(kept_mask)
        target_min = int(max(1, np.ceil(min_total / K)))
        cap = int(min(max_count, np.ceil(target_min * (1.0 + max_imbalance))))
        print("Warning: No weeks met the threshold; including all weeks with relaxed thresholds.")

    # Final sampling per kept week
    kept_counts = nonzero_counts.loc[kept_mask]
    target_min = int(np.ceil(min_total / K))
    cap = int(min(max_count, np.ceil(target_min * (1.0 + max_imbalance))))

    parts = []
    for wk_value, group in df.groupby('_week'):
        if wk_value not in kept_mask:
            continue
        n = min(len(group), cap)
        # Since kept weeks have counts >= target_min, n >= target_min is ensured up to cap
        if n < len(group):
            idx = rng.choice(group.index.values, size=n, replace=False)
            sampled = group.loc[idx]
        else:
            sampled = group
        parts.append(sampled)

    if not parts:
        print("No data sampled after applying thresholds; returning original data without changes.")
        balanced = df.drop(columns=['_week'])
    else:
        balanced = pd.concat(parts, ignore_index=True).drop(columns=['_week'])

    # Report distribution after sampling
    post_counts = balanced.groupby(robust_to_datetime(balanced[dt_col]).dt.to_period(f"W-{week_anchor}"))\
                           .size().sort_index()
    if len(post_counts) > 0:
        post_min = int(post_counts.min())
        post_max = int(post_counts.max())
        ratio = (post_max / post_min) if post_min > 0 else float('inf')
        print(f"Weekly counts after sampling: weeks={len(post_counts)}, min={post_min}, max={post_max}, max/min ratio={ratio:.3f}")

    return balanced


def plot_distributions(before_counts: pd.Series, after_counts: pd.Series, output_path: Path):
    try:
        import matplotlib.pyplot as plt  # Lazy import
    except Exception as e:
        print(f"matplotlib not available ({e}). Skipping plot generation.")
        return

    plt.figure(figsize=(14, 8))

    # Align indices for consistent plotting
    all_weeks = before_counts.index.union(after_counts.index)
    b = before_counts.reindex(all_weeks, fill_value=0)
    a = after_counts.reindex(all_weeks, fill_value=0)

    # Plot side-by-side bars
    x = np.arange(len(all_weeks))
    width = 0.45
    plt.bar(x - width/2, b.values, width=width, label='Before', alpha=0.7)
    plt.bar(x + width/2, a.values, width=width, label='After', alpha=0.7)

    # X ticks
    labels = [str(w) for w in all_weeks]
    plt.xticks(x[::max(1, len(x)//20)], labels[::max(1, len(labels)//20)], rotation=45, ha='right')

    plt.ylabel('Tweet count')
    plt.title('Weekly tweet distribution: before vs after balancing')
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved weekly distribution plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Balance tweets by week with near-even distribution (within tolerance) while preserving at least a minimum total number of tweets.')
    parser.add_argument('--input', type=str, default=None, help='Path to input CSV. Defaults to data/... with fallback to src/sentiment_analysis/data/...')
    parser.add_argument('--output', type=str, default=None, help='Path to output CSV. By default saved next to input with balanced_weekly_ prefix.')
    parser.add_argument('--date-column', type=str, default=None, help='Name of datetime column if auto-detection fails.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--week-anchor', type=str, default='MON', help='Week anchor day (e.g., MON, SUN). Default: MON')
    parser.add_argument('--min-total', type=int, default=100000, help='Minimum total number of tweets to preserve after balancing (default: 100000).')
    parser.add_argument('--max-imbalance', type=float, default=0.20, help='Allowed relative spread between weeks (e.g., 0.20 means max <= 1.2 * min). Default: 0.20')
    # Deprecated: mode retained for backward compatibility, ignored
    parser.add_argument('--mode', type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_path = find_input_file(repo_root, Path(args.input) if args.input else None)

    if input_path is None:
        print("Could not find input CSV. Checked default locations. Provide --input explicitly.")
        sys.exit(1)

    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)

    dt_col = detect_datetime_column(df, args.date_column)
    if dt_col is None:
        print("Failed to detect a datetime column. Please provide via --date-column.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    print(f"Using datetime column: '{dt_col}'")

    rng = np.random.RandomState(args.seed)

    # Compute pre counts
    tmp = df.copy()
    tmp[dt_col] = robust_to_datetime(tmp[dt_col])
    pre_counts = tmp.dropna(subset=[dt_col]).groupby(tmp[dt_col].dt.to_period(f"W-{args.week_anchor}"))\
                   .size().sort_index()

    # Balance with constraints
    balanced = balance_by_week(
        df,
        dt_col,
        rng,
        week_anchor=args.week_anchor,
        min_total=args.min_total,
        max_imbalance=args.max_imbalance,
    )

    # Prepare outputs
    if args.output:
        output_csv = Path(args.output) if Path(args.output).is_absolute() else (repo_root / args.output)
    else:
        output_csv = input_path.parent / f"balanced_weekly_{input_path.name}"


    print(f"Writing balanced dataset: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=output_csv.stem + "_", suffix=".tmp", dir=str(output_csv.parent))
    os.close(fd)
    tmp_path = Path(tmp_path)

    balanced.to_csv(tmp_path, index=False)
    #balanced.to_csv(output_csv, index=False)

    # Compute post counts
    post_counts = balanced.groupby(robust_to_datetime(balanced[dt_col]).dt.to_period(f"W-{args.week_anchor}"))\
                          .size().sort_index()

    # Save counts CSVs as a reliable fallback/record
    counts_before_csv = output_csv.parent / f"{output_csv.stem}_weekly_counts_before.csv"
    counts_after_csv = output_csv.parent / f"{output_csv.stem}_weekly_counts_after.csv"
    pre_counts.rename('count').to_csv(counts_before_csv, header=True)
    post_counts.rename('count').to_csv(counts_after_csv, header=True)
    print(f"Saved weekly counts CSVs: {counts_before_csv.name}, {counts_after_csv.name}")

    # Plot and save distribution (if matplotlib available)
    plot_path = output_csv.parent / f"{output_csv.stem}_weekly_distribution.png"
    plot_distributions(pre_counts, post_counts, plot_path)

    print("Done.")


if __name__ == "__main__":
    main()
