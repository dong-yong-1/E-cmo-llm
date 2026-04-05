import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按方案2构建正式数据集：原始数据切分，规则数据主要并入 train。")
    parser.add_argument("--base-train", default="data/train.json", help="原始数据 train 文件")
    parser.add_argument("--base-val", default="data/val.json", help="原始数据 val 文件")
    parser.add_argument("--base-test", default="data/test.json", help="原始数据 test 文件")
    parser.add_argument("--rules-data", default="ecommerce_sft_rules_v1.json", help="规则定向数据文件")
    parser.add_argument("--output-dir", default="data/final", help="最终输出目录")
    parser.add_argument("--val-rules-ratio", type=float, default=0.1, help="规则数据中分配给 val 的比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def deduplicate(records: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for record in records:
        key = (
            record["category"],
            record["subcategory"],
            record["instruction"],
            json.dumps(record["input"], ensure_ascii=False, sort_keys=True),
            record["output"],
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def summarize(records: list[dict]) -> dict:
    return {
        "count": len(records),
        "category": dict(Counter(record["category"] for record in records)),
        "subcategory": dict(Counter(record["subcategory"] for record in records)),
        "difficulty": dict(Counter(record["difficulty"] for record in records)),
        "source": dict(Counter(record["source"] for record in records)),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    base_train = load_json(args.base_train)
    base_val = load_json(args.base_val)
    base_test = load_json(args.base_test)
    rules_data = load_json(args.rules_data)

    rules_copy = rules_data[:]
    rng.shuffle(rules_copy)

    val_rules_count = int(len(rules_copy) * args.val_rules_ratio)
    val_rules = rules_copy[:val_rules_count]
    train_rules = rules_copy[val_rules_count:]

    final_train = deduplicate(base_train + train_rules)
    final_val = deduplicate(base_val + val_rules)
    final_test = deduplicate(base_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_json(output_dir / "train.json", final_train)
    dump_json(output_dir / "val.json", final_val)
    dump_json(output_dir / "test.json", final_test)

    meta = {
        "strategy": "scheme_2_base_split_plus_rules_to_train",
        "seed": args.seed,
        "inputs": {
            "base_train": args.base_train,
            "base_val": args.base_val,
            "base_test": args.base_test,
            "rules_data": args.rules_data,
        },
        "val_rules_ratio": args.val_rules_ratio,
        "summary": {
            "train": summarize(final_train),
            "val": summarize(final_val),
            "test": summarize(final_test),
        },
    }
    dump_json(output_dir / "meta.json", meta)

    print(f"构建完成：train={len(final_train)}, val={len(final_val)}, test={len(final_test)}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
