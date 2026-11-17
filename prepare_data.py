"""Build aligned graph inputs from the official public IDAO 2022 archive, offline."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from coordinate_matching import contains_coordinate

ROOT = Path(__file__).resolve().parent
EXPECTED = "70deb90928bdb31a5db14addea5b18215b8eda1565f6ed42224f3f4ee57ef29b"


def coordinate(site):
    return tuple(np.around(site.frac_coords, 5))


def serialize(site):
    return [str(site.specie), *map(float, coordinate(site))]


def site_order(site):
    return (str(site.specie), *np.around(site.frac_coords, 8))


def reference_sites():
    lattice = Lattice.from_parameters(25.5225256, 25.5225256, 14.879004, 90, 90, 120)
    elements, coords = [], []
    for atom, x, y, z in [("S", .08333333, .04166667, .144826),
                           ("Mo", .04166667, .08333333, .25),
                           ("S", .08333333, .04166667, .355174)]:
        for i in np.linspace(x, x + .875, 8):
            for j in np.linspace(y, y + .875, 8):
                elements.append(atom)
                coords.append([i, j, z])
    return {coordinate(s): s for s in Structure(lattice, elements, coords)}


def variants(structure, host):
    actual = {coordinate(s): s for s in structure}
    if len(actual) != len(structure) or not actual.keys() <= host.keys():
        raise ValueError("Input is not an aligned subset of the reference lattice.")
    changed = [s for k, s in actual.items() if s.specie != host[k].specie]
    lost = [s for k, s in host.items() if k not in actual or s.specie != actual[k].specie]
    locations = np.asarray([coordinate(s) for s in changed]).reshape(-1, 3)
    baseline = changed + [s for s in lost if not np.isin(coordinate(s), locations, assume_unique=True).all()]
    corrected = changed + [s for s in lost if not contains_coordinate(coordinate(s), locations)]
    return [[serialize(s) for s in sorted(sites, key=site_order)] for sites in (baseline, corrected)]


def main(archive, output):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Unexpected source archive. Use the exact version in data/README.md.")
    host = reference_sites()
    rows = []
    with tarfile.open(archive) as tf:
        files = [m for m in tf.getmembers() if m.isfile()]
        target = next(m for m in files if m.name.endswith("/targets.csv"))
        raw_csv = io.StringIO(tf.extractfile(target).read().decode())
        labels = {r["_id"]: float(r["band_gap"]) for r in csv.DictReader(raw_csv)}
        for member in files:
            if not member.name.endswith(".json"):
                continue
            key = Path(member.name).stem
            a, b = variants(Structure.from_dict(json.load(tf.extractfile(member))), host)
            rows.append(dict(id=key, band_gap_ev=labels[key], changed=a != b, baseline=a, corrected=b))
    if len(rows) != 2966 or sum(r["changed"] for r in rows) != 14:
        raise ValueError("Unexpected structure or changed-input count.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as stream:
        for row in sorted(rows, key=lambda r: r["id"]):
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Prepared {len(rows)} structures; 14 differ between variants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/structures.jsonl.gz")
    args = parser.parse_args()
    main(args.archive, args.output)
