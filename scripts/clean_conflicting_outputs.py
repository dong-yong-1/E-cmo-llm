from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from dataset_schema import validate_record


SAFE_PRODUCT_RULE = (
    "当上下文未提供明确商品属性时，应以商品详情页为准或说明可进一步核实，"
    "不得编造成分、材质、库存或适用人群。"
)
PENDING_URGE_RULE = "订单待发货时可帮助催仓，但无法保证立即发出。"
IN_TRANSIT_URGE_RULE = "订单运输中无法人工加急，只能持续关注物流轨迹，不得承诺已开启紧急处理。"
PENDING_ADDRESS_RULE = "订单待发货时可尝试修改地址，但修改结果需以仓库拦截或系统处理结果为准。"
SHIPPED_ADDRESS_RULE = "订单发货后通常无法直接修改地址，可建议联系派件员或快递协商，不得承诺已联系仓库改址。"
REFUND_OPENED_RULE = "商品已拆封且非质量问题时，通常不支持7天无理由退货。"
REFUND_UNOPENED_RULE = "商品未拆封且不影响二次销售，支持7天无理由退货。"
QUALITY_REFUND_RULE = "商品存在质量问题时可申请退货退款，并由商家承担相关运费。"
EXCHANGE_RULE = "换货或补发通常需要用户先在订单页面提交售后申请，审核通过后再安排处理，不得承诺固定联系时限。"
COMPLAINT_RULE = "投诉场景需先安抚情绪，再给出明确处理路径；不得承诺具体反馈时限、电话短信通知或已安排专员。"


def dedupe_tags(tags: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if tag not in seen:
            result.append(tag)
            seen.add(tag)
    return result


def audience_output(product_category: str) -> str:
    return (
        f"您好，这款{product_category}的具体适用人群建议您优先参考商品详情页中的商品说明或参数介绍。"
        "如果页面信息还不够清楚，您可以把商品链接发给我，我帮您进一步核实。"
    )


def stock_output() -> str:
    return (
        "您好，商品库存是实时变动的，建议您以当前商品详情页显示的信息为准。"
        "如果您需要进一步确认，可以把商品名称或链接发给我，我帮您核实。"
    )


def material_output(product_category: str) -> str:
    if product_category == "食品":
        return (
            "您好，这款食品的具体配料信息建议您以商品详情页或包装上的配料表为准。"
            "如果您需要进一步确认，可以把商品链接发给我，我帮您核实。"
        )
    return (
        f"您好，这款{product_category}的具体材质信息建议您优先查看商品详情页中的参数或材质说明。"
        "如果页面信息不够清楚，您可以把商品名称或链接发给我，我帮您进一步核实。"
    )


def warranty_output() -> str:
    return (
        "您好，具体保修时长和范围建议您优先查看商品详情页或售后说明。"
        "如果页面信息不够清楚，您可以把商品名称或链接发给我，我帮您进一步核实。"
    )


def pending_urge_output() -> str:
    return (
        "您好，您的订单当前还是待发货状态，我们可以帮您催仓，但暂时无法保证具体发出时间。"
        "发货后物流信息会更新，您可以在订单页面持续关注。"
    )


def in_transit_urge_output() -> str:
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
        "您好，订单发货后通常无法直接在后台修改地址。"
        "建议您先联系快递员或快递公司尝试协商转寄，具体结果需以物流方反馈为准。"
    )


def unopened_refund_output() -> str:
    return (
        "您好，如果商品未拆封且不影响二次销售，一般支持7天无理由退货。"
        "您可以在订单页面提交退货申请，我们会按流程为您处理。"
    )


def opened_non_quality_refund_output() -> str:
    return (
        "您好，如果商品已经拆封且没有质量问题，通常不支持7天无理由退货。"
        "如果您担心商品存在异常，也可以把订单号和具体情况发给我，我帮您进一步核实。"
    )


def quality_refund_output() -> str:
    return (
        "您好，如果商品存在质量问题，是可以申请退货退款的。"
        "请您在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快处理，相关运费由商家承担。"
    )


def exchange_output() -> str:
    return (
        "您好，如果商品存在破损或质量问题，可以申请换货或补发。"
        "请您先在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快为您安排后续处理。"
    )


def complaint_output(record: dict) -> str:
    instruction = record["instruction"]
    order_status = record["input"]["order_status"]
    has_quality_issue = record["input"]["has_quality_issue"]

    if "商家态度不好" in instruction:
        return (
            "非常抱歉给您带来了不愉快的体验。"
            "请您提供一下订单号和相关沟通记录，我们会尽快为您核实情况并持续跟进处理。"
        )
    if "客服一直不回复" in instruction:
        return (
            "非常抱歉给您带来了不好的体验。"
            "为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会先为您记录并继续跟进。"
        )
    if "质量太差了怎么办" in instruction:
        if order_status == "待发货" and not has_quality_issue:
            return (
                "非常抱歉给您带来了不好的体验。"
                "如果您目前对商品质量已有疑虑，且订单还未发货，我们可以协助您拦截订单或申请退款。"
                "您也可以把订单号发给我，我为您进一步核实并跟进处理。"
            )
        return (
            "非常抱歉给您带来了不好的体验。"
            "商品目前还在运输途中，建议您收到后先检查。"
            "如果确认存在质量问题，您可以第一时间联系我们，我们会协助您处理退换货。"
        )
    return (
        "非常抱歉给您带来了不好的体验。"
        "为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会先为您记录并继续跟进。"
    )


def clean_record(record: dict) -> tuple[dict, list[str]]:
    cleaned = deepcopy(record)
    changes: list[str] = []
    instruction = cleaned["instruction"]
    input_data = cleaned["input"]
    subcategory = cleaned["subcategory"]
    product_category = input_data["product_category"]
    order_status = input_data["order_status"]

    if subcategory == "商品咨询":
        cleaned["input"]["platform_rule"] = SAFE_PRODUCT_RULE
        cleaned["policy_tags"] = dedupe_tags(["商品咨询", "信息不足先说明"] + cleaned["policy_tags"])
        if "材质是什么" in instruction:
            new_output = material_output(product_category)
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_product_material")
        elif "库存还有吗" in instruction:
            new_output = stock_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_product_stock")
        elif "支持保修吗" in instruction:
            new_output = warranty_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_product_warranty")
        elif "适合什么人群" in instruction:
            new_output = audience_output(product_category)
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_product_audience")

    elif subcategory == "催发货":
        if order_status == "待发货":
            cleaned["input"]["platform_rule"] = PENDING_URGE_RULE
            new_output = pending_urge_output()
            cleaned["policy_tags"] = dedupe_tags(["催发货", "待发货可催仓"] + cleaned["policy_tags"])
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_pending_urge")
        else:
            if order_status not in {"运输中", "已发货"}:
                cleaned["input"]["order_status"] = "运输中"
                changes.append("fixed_urge_status_to_transit")
            cleaned["input"]["platform_rule"] = IN_TRANSIT_URGE_RULE
            cleaned["policy_tags"] = dedupe_tags(["催发货", "运输中不可加急"] + cleaned["policy_tags"])
            new_output = in_transit_urge_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_in_transit_urge")

    elif subcategory == "地址修改":
        if order_status == "待发货":
            cleaned["input"]["platform_rule"] = PENDING_ADDRESS_RULE
            cleaned["policy_tags"] = ["地址修改", "待发货可尝试改地址"]
            new_output = pending_address_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_pending_address")
        else:
            if order_status not in {"运输中", "已发货"}:
                cleaned["input"]["order_status"] = "运输中"
                changes.append("fixed_address_status_to_transit")
            cleaned["input"]["platform_rule"] = SHIPPED_ADDRESS_RULE
            cleaned["policy_tags"] = ["地址修改", "发货后不可改地址"]
            new_output = shipped_address_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_shipped_address")

    elif subcategory == "退货退款":
        if input_data["has_quality_issue"]:
            cleaned["input"]["platform_rule"] = QUALITY_REFUND_RULE
            cleaned["policy_tags"] = dedupe_tags(["退货退款", "质量问题可退"] + cleaned["policy_tags"])
            new_output = quality_refund_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_quality_refund")
        elif input_data["is_opened"]:
            cleaned["input"]["platform_rule"] = REFUND_OPENED_RULE
            cleaned["policy_tags"] = ["退货退款", "不可无理由退货"]
            new_output = opened_non_quality_refund_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_opened_non_quality_refund")
        else:
            cleaned["input"]["platform_rule"] = REFUND_UNOPENED_RULE
            cleaned["policy_tags"] = ["退货退款", "可无理由退货"]
            new_output = unopened_refund_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_unopened_refund")

    elif subcategory == "换货补发":
        cleaned["input"]["platform_rule"] = EXCHANGE_RULE
        cleaned["policy_tags"] = dedupe_tags(["换货补发", "先提交售后"] + cleaned["policy_tags"])
        if input_data["has_quality_issue"] or "坏了" in instruction or "换货" in instruction or "补发" in instruction:
            new_output = exchange_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_exchange_flow")

    elif subcategory == "投诉安抚":
        cleaned["input"]["platform_rule"] = COMPLAINT_RULE
        cleaned["policy_tags"] = dedupe_tags(["投诉安抚", "先安抚再处理"] + cleaned["policy_tags"])
        new_output = complaint_output(cleaned)
        if cleaned["output"] != new_output:
            cleaned["output"] = new_output
            changes.append("normalized_complaint_flow")

    return cleaned, changes


def process_split(path: Path, clean_split: bool) -> tuple[list[dict], Counter]:
    records = json.loads(path.read_text())
    if not clean_split:
        return records, Counter()

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
    parser = argparse.ArgumentParser(description="Clean conflicting signals from e-commerce CS datasets.")
    parser.add_argument("--input-dir", default="data/final_v3")
    parser.add_argument("--output-dir", default="data/final_v4_clean")
    parser.add_argument("--clean-test", action="store_true", help="Whether to clean test split too.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counters: dict[str, dict[str, int]] = {}
    total_counter: Counter = Counter()

    for split in ["train", "val", "test"]:
        clean_split = split != "test" or args.clean_test
        cleaned_records, counter = process_split(input_dir / f"{split}.json", clean_split=clean_split)
        (output_dir / f"{split}.json").write_text(
            json.dumps(cleaned_records, ensure_ascii=False, indent=2) + "\n"
        )
        split_counters[split] = dict(counter)
        total_counter.update(counter)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "clean_test": args.clean_test,
        "split_changes": split_counters,
        "total_changes": dict(total_counter),
    }
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
