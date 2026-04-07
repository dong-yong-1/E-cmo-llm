from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from dataset_schema import validate_record


PENDING_URGE_RULE = "订单待发货时可帮助催仓，但无法保证立即发出，也不能承诺当天送达。"
IN_TRANSIT_URGE_RULE = "订单运输中或已发货后无法人工加急、优先派送或承诺当天送达，只能持续关注物流轨迹。"
PENDING_ADDRESS_RULE = "订单待发货时可尝试修改地址，但修改结果需以仓库拦截或系统处理结果为准。"
SHIPPED_ADDRESS_RULE = "订单发货后通常无法直接修改地址，可建议联系派件员或快递协商，不得承诺后台改址成功。"
WRONG_ITEM_RULE = "发错货时应引导用户先在订单页面提交售后申请并上传凭证，审核通过后安排换货或补发，不得承诺专员联系或即时处理。"
DAMAGED_ITEM_RULE = "商品破损或质量异常时应引导用户提交售后申请并上传凭证，审核通过后安排换货或补发，不得承诺固定反馈时限。"
GENERIC_EXCHANGE_RULE = "换货或补发通常需要用户先在订单页面提交售后申请，审核通过后再安排处理，不得承诺固定联系时限。"


def dedupe_tags(tags: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if tag not in seen:
            result.append(tag)
            seen.add(tag)
    return result


def pending_urge_output(instruction: str) -> str:
    if "今天" in instruction or "优先送到" in instruction:
        return (
            "您好，您的订单当前还是待发货状态，我们可以帮您催仓，但暂时无法保证今天发出或优先送达。"
            "发货后物流信息会更新，您可以在订单页面持续关注。"
        )
    return (
        "您好，您的订单当前还是待发货状态，我们可以帮您催仓，但暂时无法保证具体发出时间。"
        "发货后物流信息会更新，您可以在订单页面持续关注。"
    )


def in_transit_urge_output(instruction: str) -> str:
    if "今天" in instruction or "优先送到" in instruction:
        return (
            "您好，您的订单已经在运输途中，目前无法再人工加急或承诺今天优先送达。"
            "建议您先关注订单页面的物流轨迹，如后续出现异常，我们也会继续协助您跟进。"
        )
    return (
        "您好，您的订单已经在运输途中，目前无法再人工加急。"
        "建议您先关注订单页面的物流轨迹，如后续出现异常，我们也会继续协助您跟进。"
    )


def pending_address_output() -> str:
    return (
        "您好，当前订单还未发货，您可以先尝试在订单页面修改收货地址。"
        "如果页面暂时无法操作，我们也可以帮您进一步核实处理，但最终仍需以仓库拦截结果为准。"
    )


def shipped_address_output() -> str:
    return (
        "您好，订单发货后通常无法直接修改收货地址。"
        "建议您先联系快递员或快递公司尝试协商转寄，具体结果需以物流方反馈为准。"
    )


def wrong_item_output() -> str:
    return (
        "非常抱歉给您带来了不便。如果您收到的是错误商品，请先在订单页面提交售后申请，"
        "并上传商品照片或相关凭证。审核通过后，我们会尽快为您安排换货或补发正确商品。"
    )


def damaged_item_output() -> str:
    return (
        "非常抱歉给您带来了不好的体验。如果商品存在破损或质量异常，请您先在订单页面提交售后申请，"
        "并上传相关照片或凭证。审核通过后，我们会尽快为您安排换货或补发处理。"
    )


def generic_exchange_output() -> str:
    return (
        "您好，如需换货或补发，请您先在订单页面提交售后申请并说明具体原因。"
        "审核通过后，我们会按流程为您安排后续处理。"
    )


def clean_record(record: dict) -> tuple[dict, list[str]]:
    cleaned = deepcopy(record)
    changes: list[str] = []
    instruction = cleaned["instruction"]
    input_data = cleaned["input"]
    subcategory = cleaned["subcategory"]
    order_status = input_data["order_status"]

    if subcategory == "催发货":
        if order_status == "待发货":
            cleaned["input"]["platform_rule"] = PENDING_URGE_RULE
            cleaned["policy_tags"] = dedupe_tags(["催发货", "待发货可催仓", "不可承诺发货时效"] + cleaned["policy_tags"])
            new_output = pending_urge_output(instruction)
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_urge_pending_v2")
        else:
            if order_status not in {"运输中", "已发货"}:
                cleaned["input"]["order_status"] = "运输中"
                changes.append("fixed_urge_status_to_transit_v2")
            cleaned["input"]["platform_rule"] = IN_TRANSIT_URGE_RULE
            cleaned["policy_tags"] = dedupe_tags(["催发货", "运输中不可加急", "不可承诺优先送达"] + cleaned["policy_tags"])
            new_output = in_transit_urge_output(instruction)
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_urge_transit_v2")

    elif subcategory == "地址修改":
        if order_status == "待发货":
            cleaned["input"]["platform_rule"] = PENDING_ADDRESS_RULE
            cleaned["policy_tags"] = ["地址修改", "待发货可尝试改地址"]
            new_output = pending_address_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_address_pending_v2")
        else:
            if order_status not in {"运输中", "已发货"}:
                cleaned["input"]["order_status"] = "运输中"
                changes.append("fixed_address_status_to_transit_v2")
            cleaned["input"]["platform_rule"] = SHIPPED_ADDRESS_RULE
            cleaned["policy_tags"] = ["地址修改", "发货后不可改地址"]
            new_output = shipped_address_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_address_transit_v2")

    elif subcategory == "换货补发":
        if "发错货" in instruction:
            cleaned["input"]["platform_rule"] = WRONG_ITEM_RULE
            cleaned["policy_tags"] = dedupe_tags(["换货补发", "发错货", "先提交售后"] + cleaned["policy_tags"])
            new_output = wrong_item_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_wrong_item_exchange_v2")
        elif input_data["has_quality_issue"] or any(token in instruction for token in ("坏了", "破损")):
            cleaned["input"]["platform_rule"] = DAMAGED_ITEM_RULE
            cleaned["policy_tags"] = dedupe_tags(["换货补发", "质量问题可换货", "先提交售后"] + cleaned["policy_tags"])
            new_output = damaged_item_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_damaged_exchange_v2")
        else:
            cleaned["input"]["platform_rule"] = GENERIC_EXCHANGE_RULE
            cleaned["policy_tags"] = dedupe_tags(["换货补发", "先提交售后"] + cleaned["policy_tags"])
            new_output = generic_exchange_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_generic_exchange_v2")

    return cleaned, changes


def process_split(path: Path) -> tuple[list[dict], Counter]:
    records = json.loads(path.read_text())
    cleaned_records: list[dict] = []
    counter: Counter = Counter()
    for record in records:
        cleaned, changes = clean_record(record)
        validate_record(cleaned)
        cleaned_records.append(cleaned)
        for change in changes:
            counter[change] += 1
    return cleaned_records, counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Stabilize weak e-commerce subtasks with consistent supervision.")
    parser.add_argument("--input-dir", default="data/final_v4_clean")
    parser.add_argument("--output-dir", default="data/final_v5_stable")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counters: dict[str, dict[str, int]] = {}
    total_counter: Counter = Counter()

    for split in ("train", "val", "test"):
        cleaned_records, counter = process_split(input_dir / f"{split}.json")
        (output_dir / f"{split}.json").write_text(
            json.dumps(cleaned_records, ensure_ascii=False, indent=2) + "\n"
        )
        split_counters[split] = dict(counter)
        total_counter.update(counter)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "split_changes": split_counters,
        "total_changes": dict(total_counter),
    }
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
