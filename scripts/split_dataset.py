import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 schema v1 数据集切分为 train/val/test 三份。")
    parser.add_argument("--input", default="ecommerce_sft_v1.json", help="输入数据集路径")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--stratify-by",
        default="subcategory,difficulty",
        help="分层字段，多个字段用英文逗号分隔",
    )
    return parser.parse_args()


def ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train/val/test 比例之和必须为 1，当前为 {total}")


def bucket_key(record: dict, stratify_fields: list[str]) -> tuple:
    return tuple(record[field] for field in stratify_fields)


def split_bucket(records: list[dict], train_ratio: float, val_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(records)
    train_count = int(n * train_ratio)
    val_count = int(n * val_ratio)
    test_count = n - train_count - val_count

    # 尽量保证每个分桶都有 val/test，避免正式评估缺失某些场景
    if n >= 3:
        if val_count == 0:
            val_count = 1
            train_count = max(1, train_count - 1)
        test_count = n - train_count - val_count
        if test_count == 0:
            test_count = 1
            train_count = max(1, train_count - 1)
            val_count = n - train_count - test_count

    train_split = records[:train_count]
    val_split = records[train_count:train_count + val_count]
    test_split = records[train_count + val_count:]
    return train_split, val_split, test_split


def summarize(records: list[dict], stratify_fields: list[str]) -> dict:
    summary = {
        "count": len(records),
        "category": dict(Counter(record["category"] for record in records)),
        "subcategory": dict(Counter(record["subcategory"] for record in records)),
        "difficulty": dict(Counter(record["difficulty"] for record in records)),
    }
    stratify_counter = Counter(bucket_key(record, stratify_fields) for record in records)
    summary["stratify_buckets"] = {"|".join(map(str, key)): count for key, count in stratify_counter.items()}
    return summary


def main() -> None:
    args = parse_args()
    ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    stratify_fields = [field.strip() for field in args.stratify_by.split(",") if field.strip()]
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(args.seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in data:
        buckets[bucket_key(record, stratify_fields)].append(record)

    train_data: list[dict] = []
    val_data: list[dict] = []
    test_data: list[dict] = []

    for _, records in buckets.items():
        records_copy = records[:]
        rng.shuffle(records_copy)
        train_split, val_split, test_split = split_bucket(records_copy, args.train_ratio, args.val_ratio)
        train_data.extend(train_split)
        val_data.extend(val_split)
        test_data.extend(test_split)

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    output_files = {
        "train.json": train_data,
        "val.json": val_data,
        "test.json": test_data,
    }

    for filename, records in output_files.items():
        with (output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    split_meta = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "stratify_fields": stratify_fields,
        "summary": {
            "train": summarize(train_data, stratify_fields),
            "val": summarize(val_data, stratify_fields),
            "test": summarize(test_data, stratify_fields),
        },
    }

    with (output_dir / "split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    print(f"切分完成：train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
