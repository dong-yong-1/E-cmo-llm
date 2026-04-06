from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge targeted data into cleaned train/val splits.")
    parser.add_argument("--base-dir", default="data/final_clean")
    parser.add_argument("--targeted-path", default="data/targeted/overpromise_targeted_v1.json")
    parser.add_argument("--output-dir", default="data/final_v2")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    train = load_json(base_dir / "train.json")
    val = load_json(base_dir / "val.json")
    test = load_json(base_dir / "test.json")
    targeted = load_json(Path(args.targeted_path))

    rng = random.Random(args.seed)
    targeted = targeted[:]
    rng.shuffle(targeted)

    val_size = max(1, int(len(targeted) * args.val_ratio))
    targeted_val = targeted[:val_size]
    targeted_train = targeted[val_size:]

    merged_train = train + targeted_train
    merged_val = val + targeted_val

    write_json(output_dir / "train.json", merged_train)
    write_json(output_dir / "val.json", merged_val)
    write_json(output_dir / "test.json", test)

    meta = {
        "base_dir": str(base_dir),
        "targeted_path": args.targeted_path,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "counts": {
            "base_train": len(train),
            "base_val": len(val),
            "base_test": len(test),
            "targeted_total": len(targeted),
            "targeted_train": len(targeted_train),
            "targeted_val": len(targeted_val),
            "merged_train": len(merged_train),
            "merged_val": len(merged_val),
            "merged_test": len(test),
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
