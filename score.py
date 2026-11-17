"""Score the six saved pairs; needs no packages outside the standard library."""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
EXPECTED = {(split, seed) for split in (101, 202, 303) for seed in (11, 29)}


def summarize(path):
    grouped = defaultdict(list)
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[(int(row["split"]), int(row["seed"]))].append(row)
    if set(grouped) != EXPECTED:
        raise ValueError("Expected three splits and two initialization seeds.")
    summaries = []
    for (split, seed), rows in sorted(grouped.items()):
        if len(rows) != 467 or {int(r["test_position"]) for r in rows} != set(range(467)):
            raise ValueError("Expected 467 distinct test positions per pair.")
        summary = {"split": split, "seed": seed}
        for variant in ("baseline", "corrected"):
            errors = [float(r[variant + "_abs_error_ev"]) for r in rows]
            summary[variant + "_ewt_pct"] = 100 * mean(e < 0.02 for e in errors)
            summary[variant + "_mae_mev"] = 1000 * mean(errors)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/test-errors.csv.gz")
    parser.add_argument("--csv", type=Path, help="Optionally export the six-pair table.")
    args = parser.parse_args()
    table = summarize(args.input)
    if args.csv:
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print("Split  Seed    EwT before → after (%)     MAE before → after (meV)")
    for r in table:
        print(f"{r['split']:5} {r['seed']:5}    {r['baseline_ewt_pct']:6.2f} → {r['corrected_ewt_pct']:6.2f}"
              f"            {r['baseline_mae_mev']:6.2f} → {r['corrected_mae_mev']:6.2f}")
    agg = {key: mean(r[key] for r in table) for key in list(table[0])[2:]}
    print(f"Mean: EwT {agg['baseline_ewt_pct']:.2f} → {agg['corrected_ewt_pct']:.2f}%; "
          f"MAE {agg['baseline_mae_mev']:.2f} → {agg['corrected_mae_mev']:.2f} meV.")
