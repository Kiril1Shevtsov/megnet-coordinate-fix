"""Recreate the training protocol or verify one bundled MEGNet checkpoint."""
import argparse
import csv
import gzip
import json
import os
import random
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTEROP_THREADS="1",
                  TF_NUM_INTRAOP_THREADS="1", OMP_NUM_THREADS="1",
                  TF_DETERMINISTIC_OPS="1", TF_CUDNN_DETERMINISTIC="1")

import numpy as np
import tensorflow as tf
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parent
SPLITS, SEEDS = (101, 202, 303), (11, 29)
VARIANTS = ("baseline", "corrected")
LATTICE = Lattice.from_parameters(25.5225256, 25.5225256, 14.879004, 90, 90, 120)


def build_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return MEGNetModel(graph_converter=CrystalGraph(cutoff=20),
                       centers=np.linspace(0, 21, 100), width=0.5)


def load_structures():
    with gzip.open(ROOT / "data/structures.jsonl.gz", "rt") as stream:
        rows = sorted(map(json.loads, stream), key=lambda r: r["id"])
    if len(rows) != 2966 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unexpected structure count or duplicate material IDs.")
    return rows


def split_rows(rows, split):
    perm = np.random.RandomState(split).permutation(len(rows))
    return [[rows[i] for i in idx] for idx in (perm[:2060], perm[2060:2499], perm[2499:])]


def make_graphs(model, rows, variant):
    graphs = []
    for row in rows:
        sites = row[variant]
        structure = Structure(LATTICE, [s[0] for s in sites], [s[1:] for s in sites])
        graphs.append(model.graph_converter.convert(structure))
    return graphs


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        path = work / f"initial-{seed}.weights.h5"
        if path.exists():
            raise FileExistsError(f"Initial state already exists: {path}")
        model = build_model(seed)
        model.save_weights(str(path))
        tf.keras.backend.clear_session()
    print(f"Initial weights ready: {work}")


def train(work, split, seed, variant):
    initial = work / f"initial-{seed}.weights.h5"
    if not initial.exists():
        raise FileNotFoundError("Run 'prepare' first.")
    target = work / f"split-{split}_seed-{seed}" / variant
    target.mkdir(parents=True, exist_ok=False)
    model = build_model(seed)
    model.load_weights(str(initial))
    if int(model.model.optimizer.iterations.numpy()) != 0:
        raise ValueError("Optimizer must start from iteration zero.")
    train_rows, valid_rows, test_rows = split_rows(load_structures(), split)
    graphs = make_graphs(model, train_rows, variant)
    labels = [model.target_scaler.transform(row["band_gap_ev"], len(graph["atom"]))
              for row, graph in zip(train_rows, graphs)]
    generator = model._create_generator(*model.graph_converter.get_flat_data(graphs, labels),
                                        batch_size=128, is_shuffle=False)
    rng = np.random.RandomState(split * 1000 + seed)
    for epoch in range(120):
        generator.mol_index = rng.permutation(2060)
        model.model.reset_metrics()
        for batch in range(len(generator)):
            inputs, response = generator[batch]
            model.model.train_on_batch(inputs, response, reset_metrics=False)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/120", flush=True)
    model.save_weights(str(target / "final.weights.h5"))
    for role, rows in [("validation", valid_rows), ("test", test_rows)]:
        pred = np.asarray(model.predict_graphs(make_graphs(model, rows, variant))).reshape(-1)
        with (target / f"{role}-predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["material_id", "target_ev", "prediction_ev"])
            writer.writerows((r["id"], r["band_gap_ev"], float(v)) for r, v in zip(rows, pred))
    print(f"Saved: {target}")


def verify(split, seed, variant):
    model = build_model(seed)
    path = ROOT / "weights" / f"split-{split}_seed-{seed}" / f"{variant}.weights.h5"
    model.load_weights(str(path))
    _, _, rows = split_rows(load_structures(), split)
    with gzip.open(ROOT / "results/test-errors.csv.gz", "rt", newline="") as stream:
        ref = [r for r in csv.DictReader(stream) if int(r["split"]) == split and int(r["seed"]) == seed]
    assert [int(r["test_position"]) for r in ref] == list(range(467)) and len(rows) == 467
    pred = np.asarray(model.predict_graphs(make_graphs(model, rows, variant))).reshape(-1)
    expected = np.asarray([float(r[f"{variant}_abs_error_ev"]) for r in ref])
    y = np.asarray([r["band_gap_ev"] for r in rows])
    observed = np.abs(pred - y)
    largest = float(np.max(np.abs(observed - expected)))
    same = np.array_equal(observed < 0.02, expected < 0.02)
    if not (largest < 1e-6 and same):
        raise RuntimeError(f"Replay mismatch: max difference={largest:g} eV, same EwT flags={same}")
    print(f"{split}/{seed}/{variant}: 467 absolute errors verified; max |delta| {largest:.3g} eV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT / "runs")
    cmds = parser.add_subparsers(dest="cmd", required=True)
    cmds.add_parser("prepare", help="Generate paired starting weights for training")
    for cmd in ["train", "verify"]:
        sub = cmds.add_parser(cmd)
        sub.add_argument("--split", type=int, choices=SPLITS, required=True)
        sub.add_argument("--seed", type=int, choices=SEEDS, required=True)
        sub.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.workdir)
    elif args.cmd == "train":
        train(args.workdir, args.split, args.seed, args.variant)
    else:
        verify(args.split, args.seed, args.variant)
