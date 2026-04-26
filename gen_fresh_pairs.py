#!/usr/bin/env python3
"""
gen_fresh_pairs.py — Generate fresh arc-gen pairs and test your ONNX models against them.

Usage:
    # Generate fresh pairs for all tasks and test your submission:
    python gen_fresh_pairs.py --onnx-dir /path/to/submission --n 50

    # Just generate fresh pairs for specific tasks (no ONNX testing):
    python gen_fresh_pairs.py --tasks 18 66 76 --n 100 --no-onnx

    # Generate and save fresh pairs as JSON (same format as task JSONs):
    python gen_fresh_pairs.py --tasks 18 --n 50 --save-json
"""

import json, random, sys, argparse, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import task_list
import numpy as np

# ── ARC-GEN constants ──────────────────────────────────────────────────────
FRESH_SEED_START = 10000   # seeds 2025..2286 are already in the task JSONs
NEUROGOLF_DIR    = Path("/home/nnmax/Desktop/kaggle/neuro-golf/data/raw/neurogolf-2026")
SUBMISSION_DIR   = Path("../submission")

CHANNELS, HEIGHT, WIDTH = 10, 30, 30

# ── helpers ────────────────────────────────────────────────────────────────
def build_mapping():
    """Map neurogolf task numbers → ARC-GEN hex IDs using first train input."""
    tl = task_list.task_list()
    index = {}
    for hex_id, (gen, validator) in tl.items():
        try:
            orig = validator()
            if orig.get("train"):
                k = str(orig["train"][0]["input"])
                index[k] = (hex_id, gen)
        except Exception:
            pass

    mapping = {}
    for f in sorted(NEUROGOLF_DIR.glob("task*.json")):
        task_num = int(f.stem.replace("task", ""))
        ng = json.loads(f.read_text())
        if ng.get("train"):
            k = str(ng["train"][0]["input"])
            if k in index:
                mapping[task_num] = index[k]
    return mapping


def get_existing_inputs(task_num):
    """All input grids already seen for this task."""
    f = NEUROGOLF_DIR / f"task{task_num:03d}.json"
    ng = json.loads(f.read_text())
    seen = set()
    for split in ("train", "test", "arc-gen"):
        for p in ng.get(split, []):
            seen.add(str(p["input"]))
    return seen


def generate_fresh_pairs(gen, existing_inputs, n, max_seeds=5000):
    """Generate n pairs whose inputs were never seen before."""
    pairs = []
    for i in range(max_seeds):
        if len(pairs) >= n:
            break
        random.seed(FRESH_SEED_START + i)
        try:
            pair = gen()
            if str(pair["input"]) not in existing_inputs:
                pairs.append(pair)
        except Exception:
            pass
    return pairs


def grid_to_tensor(grid):
    t = np.zeros([1, CHANNELS, HEIGHT, WIDTH], dtype=np.float32)
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    for r in range(min(h, HEIGHT)):
        for c in range(min(w, WIDTH)):
            col = int(arr[r, c])
            if 0 <= col < CHANNELS:
                t[0, col, r, c] = 1.0
    return t


def tensor_to_grid(t):
    out = []
    for r in range(HEIGHT):
        row = []
        for c in range(WIDTH):
            cols = [i for i in range(CHANNELS) if t[0, i, r, c] > 0]
            row.append(cols[0] if len(cols) == 1 else (11 if cols else 10))
        while row and row[-1] == 10: row.pop()
        if row: out.append(row)
    while out and not out[-1]: out.pop()
    return out


def grids_match(a, b):
    if a == b: return True
    max_r = max(len(a), len(b))
    max_c = max((max(len(r) for r in a) if a else 0),
                (max(len(r) for r in b) if b else 0))
    def get(g, r, c):
        return g[r][c] if r < len(g) and c < len(g[r]) else 0
    return all(get(a,r,c) == get(b,r,c) for r in range(max_r) for c in range(max_c))


def test_model(onnx_path, pairs):
    """Returns (passed, total) for a model against fresh pairs."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        passed = 0
        for p in pairs:
            try:
                pred = sess.run(["output"], {"input": grid_to_tensor(p["input"])})[0]
                if grids_match(tensor_to_grid(pred), p["output"]):
                    passed += 1
            except Exception:
                pass
        return passed, len(pairs)
    except Exception as e:
        return None, len(pairs)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=int, help="task numbers (default: all 400)")
    parser.add_argument("--n", type=int, default=50, help="fresh pairs per task")
    parser.add_argument("--onnx-dir", type=str, default=str(SUBMISSION_DIR))
    parser.add_argument("--no-onnx", action="store_true", help="skip ONNX testing")
    parser.add_argument("--save-json", action="store_true", help="save fresh pairs as JSON")
    args = parser.parse_args()

    print("Building task mapping...")
    mapping = build_mapping()
    print(f"Mapped {len(mapping)}/400 tasks to ARC-GEN generators")

    task_nums = args.tasks or sorted(mapping.keys())
    onnx_dir  = Path(args.onnx_dir)

    results = {}
    likely_overfitted = []

    for task_num in task_nums:
        if task_num not in mapping:
            print(f"task{task_num:03d}: no ARC-GEN mapping, skipping")
            continue

        hex_id, gen = mapping[task_num]
        existing    = get_existing_inputs(task_num)
        pairs       = generate_fresh_pairs(gen, existing, args.n)

        if not pairs:
            print(f"task{task_num:03d}: exhausted combinatorial space, skipping")
            continue

        print(f"task{task_num:03d} ({hex_id}): {len(pairs)} fresh pairs", end="")

        if args.save_json:
            out = Path(f"fresh_pairs/task{task_num:03d}_fresh.json")
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps({"fresh": pairs}, indent=2))

        if not args.no_onnx:
            onnx_path = onnx_dir / f"task{task_num:03d}.onnx"
            if not onnx_path.exists():
                print(f"  [no onnx file]")
                continue
            passed, total = test_model(onnx_path, pairs)
            if passed is None:
                print(f"  [onnx error]")
                continue
            rate = passed / total if total else 0
            flag = " ← OVERFITTED?" if rate < 0.5 else ""
            print(f"  pass={passed}/{total} ({rate:.0%}){flag}")
            results[task_num] = {"hex_id": hex_id, "passed": passed, "total": total, "rate": rate}
            if rate < 0.5:
                likely_overfitted.append(task_num)
        else:
            print()

    if results:
        # Summary
        print("\n" + "="*60)
        print("OVERFITTING REPORT")
        print("="*60)
        print(f"Likely overfitted ({len(likely_overfitted)} tasks, <50% on fresh pairs):")
        for t in sorted(likely_overfitted):
            r = results[t]
            print(f"  task{t:03d}: {r['passed']}/{r['total']} ({r['rate']:.0%})")

        Path("fresh_pair_results.json").write_text(json.dumps(results, indent=2))
        print("\nFull results saved to fresh_pair_results.json")


if __name__ == "__main__":
    main()
