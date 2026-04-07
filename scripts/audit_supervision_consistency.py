from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_SUBCATEGORIES = ("催发货", "地址修改", "换货补发")


GENERIC_RISK_PATTERNS = {
    "承诺24小时反馈": r"24小时|48小时",
    "承诺电话短信联系": r"电话|短信|回电|来电",
    "承诺专员跟进": r"专员|高级客服|售后专员",
    "声称已提交或已处理": r"已为您提交|已经帮您提交|已帮您提交|已经为您提交|已为您处理|已经为您处理",
}

SUBCATEGORY_RISK_PATTERNS = {
    "催发货": {
        "承诺已联系仓库": r"联系仓库|仓库加急|催仓加急|优先发货",
        "承诺已加急": r"已加急|加急处理|紧急处理|优先配送",
        "虚构重新打包": r"重新打包|尽快发出",
    },
    "地址修改": {
        "承诺直接改址": r"已经修改地址|已帮您修改地址|直接修改地址成功|后台已经改好",
        "错误引导页面修改": r"订单页面.*修改地址|页面.*直接修改地址",
        "虚构联系仓库改址": r"联系仓库.*改地址|仓库.*修改地址",
    },
    "换货补发": {
        "承诺直接补发换货": r"直接补发|立即补发|马上补发|直接换货|立即换货",
        "额外赔偿承诺": r"赔偿|补偿|退款并补发",
    },
}


def load_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "subcategory": record["subcategory"],
        "difficulty": record["difficulty"],
        "instruction": record["instruction"],
        "order_status": record["input"]["order_status"],
        "has_quality_issue": record["input"]["has_quality_issue"],
        "platform_rule": record["input"]["platform_rule"],
        "policy_tags": record["policy_tags"],
        "output": record["output"],
    }


def check_specific_conflicts(record: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    subcategory = record["subcategory"]
    output = record["output"]
    order_status = record["input"]["order_status"]

    for label, pattern in GENERIC_RISK_PATTERNS.items():
        if re.search(pattern, output):
            issues.append({"type": "generic_risk", "label": label})

    for label, pattern in SUBCATEGORY_RISK_PATTERNS.get(subcategory, {}).items():
        if re.search(pattern, output):
            issues.append({"type": "subcategory_risk", "label": label})

    if subcategory == "催发货":
        if order_status in {"运输中", "已发货"} and re.search(r"催仓", output):
            issues.append({"type": "state_mismatch", "label": "运输中却使用催仓表述"})
        if order_status == "待发货" and re.search(r"无法再人工加急|物流轨迹", output):
            issues.append({"type": "state_mismatch", "label": "待发货却使用运输中表述"})

    if subcategory == "地址修改":
        if order_status in {"运输中", "已发货"} and (
            re.search(r"可以在订单页面修改|可在订单页面修改|支持在订单页面修改", output)
            or re.search(r"可以在后台修改|可在后台修改|后台可以修改", output)
            or re.search(r"可以直接修改地址|可直接修改地址|已经修改地址", output)
        ):
            issues.append({"type": "state_mismatch", "label": "发货后却暗示可页面或后台改址"})
        if order_status == "待发货" and re.search(r"联系快递员|快递公司|转寄", output):
            issues.append({"type": "state_mismatch", "label": "待发货却使用发货后协商话术"})

    if subcategory == "换货补发":
        if not re.search(r"售后申请|订单页面提交|审核通过", output):
            issues.append({"type": "flow_missing", "label": "缺少售后申请或审核流程"})
        if re.search(r"24小时|电话|短信|专员", output):
            issues.append({"type": "overpromise", "label": "换货补发出现额外时限或转接承诺"})

    return issues


def group_instruction_variants(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["subcategory"], normalize_text(record["instruction"]))
        buckets[key].append(record)

    variants: list[dict[str, Any]] = []
    for (subcategory, instruction_key), grouped in buckets.items():
        order_statuses = sorted({r["input"]["order_status"] for r in grouped})
        platform_rules = sorted({r["input"]["platform_rule"] for r in grouped})
        outputs = sorted({r["output"] for r in grouped})
        if len(order_statuses) > 1 or len(platform_rules) > 1 or len(outputs) > 1:
            variants.append(
                {
                    "subcategory": subcategory,
                    "instruction_key": instruction_key,
                    "count": len(grouped),
                    "order_statuses": order_statuses,
                    "platform_rule_count": len(platform_rules),
                    "output_count": len(outputs),
                    "sample_ids": [r["id"] for r in grouped[:10]],
                    "sample_instruction": grouped[0]["instruction"],
                }
            )
    variants.sort(key=lambda item: (item["subcategory"], -item["count"], -item["output_count"]))
    return variants


def audit_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    target_records = [r for r in records if r["subcategory"] in TARGET_SUBCATEGORIES]

    by_subcategory = Counter(r["subcategory"] for r in target_records)
    by_status = Counter((r["subcategory"], r["input"]["order_status"]) for r in target_records)
    by_difficulty = Counter((r["subcategory"], r["difficulty"]) for r in target_records)
    by_source = Counter((r["subcategory"], r["source"]) for r in target_records)
    tag_counter: Counter[tuple[str, str]] = Counter()
    issue_counter: Counter[tuple[str, str]] = Counter()
    suspicious_examples: list[dict[str, Any]] = []

    for record in target_records:
        for tag in record["policy_tags"]:
            tag_counter[(record["subcategory"], tag)] += 1
        issues = check_specific_conflicts(record)
        if issues:
            suspicious_examples.append(
                {
                    "record": record_summary(record),
                    "issues": issues,
                }
            )
            for issue in issues:
                issue_counter[(record["subcategory"], issue["label"])] += 1

    instruction_variants = group_instruction_variants(target_records)

    return {
        "total_target_records": len(target_records),
        "by_subcategory": dict(by_subcategory),
        "by_status": [
            {"subcategory": subcategory, "order_status": status, "count": count}
            for (subcategory, status), count in sorted(by_status.items())
        ],
        "by_difficulty": [
            {"subcategory": subcategory, "difficulty": difficulty, "count": count}
            for (subcategory, difficulty), count in sorted(by_difficulty.items())
        ],
        "by_source": [
            {"subcategory": subcategory, "source": source, "count": count}
            for (subcategory, source), count in sorted(by_source.items())
        ],
        "top_policy_tags": [
            {"subcategory": subcategory, "policy_tag": tag, "count": count}
            for (subcategory, tag), count in tag_counter.most_common(30)
        ],
        "issue_counts": [
            {"subcategory": subcategory, "issue": label, "count": count}
            for (subcategory, label), count in issue_counter.most_common()
        ],
        "instruction_variants": instruction_variants[:50],
        "suspicious_examples": suspicious_examples[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit supervision consistency for weak e-commerce customer service subtasks."
    )
    parser.add_argument("--input-dir", default="data/final_v4_clean")
    parser.add_argument("--output-dir", default="reports/supervision_audit")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "target_subcategories": list(TARGET_SUBCATEGORIES),
        "splits": {},
    }

    for split in ("train", "val"):
        records = load_records(input_dir / f"{split}.json")
        report["splits"][split] = audit_split(records)

    summary = {
        split: {
            "total_target_records": report["splits"][split]["total_target_records"],
            "issue_counts": report["splits"][split]["issue_counts"][:20],
            "instruction_variant_count": len(report["splits"][split]["instruction_variants"]),
            "suspicious_example_count": len(report["splits"][split]["suspicious_examples"]),
        }
        for split in ("train", "val")
    }

    (output_dir / "supervision_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "supervision_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
