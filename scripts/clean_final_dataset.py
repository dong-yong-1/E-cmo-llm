from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


SAFE_PRODUCT_RULE = (
    "当上下文未提供明确商品属性时，应以商品详情页为准或说明可进一步核实，"
    "不得编造成分、材质、库存或适用人群。"
)
PENDING_ADDRESS_RULE = "订单未发货前可尝试修改地址，但最终以仓库拦截结果为准。"
SHIPPED_ADDRESS_RULE = "订单发货后通常无法直接修改地址，可联系快递尝试协商。"
OPENED_REFUND_RULE = "拆封后非质量问题通常不支持7天无理由退货。"
QUALITY_REFUND_RULE = "商品存在质量问题时可申请退货退款，并由商家承担相关运费。"
QUALITY_EXCHANGE_RULE = "发错货或商品损坏时，可优先安排补发或换货。"


def dedupe_tags(tags: list[str]) -> list[str]:
    seen = set()
    result = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def safe_material_output(product_category: str) -> str:
    if product_category == "食品":
        return (
            "您好，这款食品的具体配料信息建议您以商品详情页或包装上的配料表为准。"
            "如果您需要进一步确认，可以把商品链接发给我，我帮您核实。"
        )
    return (
        f"您好，这款{product_category}的具体材质信息建议您优先查看商品详情页中的参数或材质说明。"
        "如果页面信息不够清楚，您可以把商品名称或链接发给我，我帮您进一步核实。"
    )


def safe_stock_output() -> str:
    return (
        "您好，商品库存是实时变动的，建议您以当前商品详情页显示的库存信息为准。"
        "如果您需要进一步确认，可以把商品名称或链接发给我，我帮您核实。"
    )


def safe_warranty_output() -> str:
    return (
        "您好，具体保修时长和范围建议您优先查看商品详情页或售后说明。"
        "如果页面信息不够清楚，您可以把商品名称或链接发给我，我帮您进一步核实。"
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
        "如果后续处理上还有问题，您也可以继续联系我们协助跟进。"
    )


def opened_non_quality_refund_output() -> str:
    return (
        "您好，如果商品已经拆封且没有质量问题，通常不支持7天无理由退货。"
        "如果您担心商品存在异常，也可以把订单信息和相关照片发给我，我帮您进一步核实。"
    )


def quality_refund_output() -> str:
    return (
        "您好，如果商品存在质量问题，是可以申请退货退款的。"
        "请您在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快处理，相关运费由商家承担。"
    )


def quality_exchange_output() -> str:
    return (
        "您好，如果商品存在质量问题，是可以申请换货的。"
        "请您提供订单号和商品问题照片，我们核实后会尽快为您安排换货或补发处理。"
    )


def complaint_pre_ship_output(order_status: str) -> str:
    if order_status == "待发货":
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


def clean_record(record: dict) -> tuple[dict, list[str]]:
    cleaned = deepcopy(record)
    changes: list[str] = []
    instruction = cleaned["instruction"]
    input_data = cleaned["input"]
    subcategory = cleaned["subcategory"]
    product_category = input_data["product_category"]

    if subcategory == "商品咨询":
        cleaned["input"]["platform_rule"] = SAFE_PRODUCT_RULE
        cleaned["policy_tags"] = dedupe_tags(
            ["商品咨询", "信息不足先说明", "product_info_unknown_do_not_hallucinate"]
        )
        if "材质是什么" in instruction:
            new_output = safe_material_output(product_category)
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_product_material_response")
        elif "库存还有吗" in instruction:
            new_output = safe_stock_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_product_stock_response")
        elif "支持保修吗" in instruction:
            new_output = safe_warranty_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_product_warranty_response")
        if changes:
            changes.append("normalized_product_rule_and_tags")

    if subcategory == "地址修改":
        if "发货后" in instruction or input_data["order_status"] in {"运输中", "已发货"}:
            if input_data["order_status"] == "待发货":
                cleaned["input"]["order_status"] = "运输中"
                changes.append("fixed_address_status_to_in_transit")
            cleaned["input"]["platform_rule"] = SHIPPED_ADDRESS_RULE
            cleaned["policy_tags"] = ["地址修改", "发货后不可改地址"]
            new_output = shipped_address_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_shipped_address_response")
        elif input_data["order_status"] == "待发货":
            cleaned["input"]["platform_rule"] = PENDING_ADDRESS_RULE
            cleaned["policy_tags"] = ["地址修改", "待发货可尝试改地址"]
            new_output = pending_address_output()
            if cleaned["output"] != new_output:
                cleaned["output"] = new_output
                changes.append("normalized_pending_address_response")

    if subcategory == "退货退款" and "已经拆封" in instruction and input_data["is_opened"] is False:
        cleaned["input"]["is_opened"] = True
        changes.append("fixed_opened_flag_from_instruction")

    if (
        subcategory == "退货退款"
        and cleaned["input"]["is_opened"]
        and not cleaned["input"]["has_quality_issue"]
        and "已经拆封" in instruction
    ):
        cleaned["input"]["platform_rule"] = OPENED_REFUND_RULE
        cleaned["policy_tags"] = ["退货退款", "不可无理由退货"]
        new_output = opened_non_quality_refund_output()
        if cleaned["output"] != new_output:
            cleaned["output"] = new_output
            changes.append("normalized_opened_non_quality_refund_response")

    if subcategory == "退货退款" and cleaned["input"]["has_quality_issue"]:
        cleaned["input"]["platform_rule"] = QUALITY_REFUND_RULE
        cleaned["policy_tags"] = dedupe_tags(["退货退款", "质量问题可退"])
        if "未拆封" in cleaned["output"]:
            cleaned["output"] = quality_refund_output()
            changes.append("removed_unopened_constraint_from_quality_refund")

    if subcategory == "换货补发" and cleaned["input"]["has_quality_issue"]:
        cleaned["input"]["platform_rule"] = QUALITY_EXCHANGE_RULE
        cleaned["policy_tags"] = dedupe_tags(["换货补发", "质量问题可换货"])
        if "未拆封" in cleaned["output"]:
            cleaned["output"] = quality_exchange_output()
            changes.append("removed_unopened_constraint_from_quality_exchange")

    if (
        subcategory == "投诉安抚"
        and "质量太差了怎么办" in instruction
        and not cleaned["input"]["has_quality_issue"]
        and cleaned["input"]["order_status"] in {"待发货", "运输中"}
    ):
        new_output = complaint_pre_ship_output(cleaned["input"]["order_status"])
        if cleaned["output"] != new_output:
            cleaned["output"] = new_output
            changes.append("normalized_pre_delivery_quality_complaint_response")

    return cleaned, changes


def process_split(path: Path) -> tuple[list[dict], Counter]:
    records = json.loads(path.read_text())
    cleaned_records = []
    counter: Counter = Counter()
    for record in records:
        cleaned, changes = clean_record(record)
        cleaned_records.append(cleaned)
        for change in changes:
            counter[change] += 1
    return cleaned_records, counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean final e-commerce CS datasets.")
    parser.add_argument("--input-dir", default="data/final")
    parser.add_argument("--output-dir", default="data/final_clean")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counters: dict[str, dict[str, int]] = {}
    aggregate = Counter()

    for split in ["train", "val", "test"]:
        cleaned_records, counter = process_split(input_dir / f"{split}.json")
        (output_dir / f"{split}.json").write_text(
            json.dumps(cleaned_records, ensure_ascii=False, indent=2) + "\n"
        )
        split_counters[split] = dict(counter)
        aggregate.update(counter)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "split_changes": split_counters,
        "total_changes": dict(aggregate),
    }
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
