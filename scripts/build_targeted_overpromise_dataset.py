from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from dataset_schema import SCHEMA_VERSION, validate_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "targeted" / "overpromise_targeted_v1.json"

PRODUCT_CATEGORIES = ["护肤品", "食品", "母婴用品", "家居用品", "数码配件", "服饰"]


def make_record(
    idx: int,
    category: str,
    subcategory: str,
    difficulty: str,
    question: str,
    product_category: str,
    order_status: str,
    user_emotion: str,
    platform_rule: str,
    output: str,
    policy_tags: list[str],
    *,
    is_opened: bool = False,
    has_quality_issue: bool = False,
) -> dict:
    record = {
        "id": f"targeted_{category}_{subcategory}_{idx:05d}",
        "category": category,
        "subcategory": subcategory,
        "instruction": f"用户问：{question}",
        "input": {
            "product_category": product_category,
            "order_status": order_status,
            "is_opened": is_opened,
            "has_quality_issue": has_quality_issue,
            "user_emotion": user_emotion,
            "platform_rule": platform_rule,
        },
        "output": output,
        "policy_tags": policy_tags,
        "quality_label": "good",
        "difficulty": difficulty,
        "source": "synthetic_targeted_overpromise",
        "schema_version": SCHEMA_VERSION,
    }
    validate_record(record)
    return record


def build_logistics_in_transit(records: list[dict], start_idx: int) -> int:
    questions = [
        "可以加急发货吗？",
        "为什么还没发货？",
        "今天能帮我优先送到吗？",
        "能不能帮我催快一点？",
    ]
    outputs = [
        "您好，看到您的订单已经在运输途中，目前无法再人工加急。建议您先关注订单页面的物流轨迹，如配送异常我们也会继续协助您跟进。",
        "您好，您的订单已经从仓库发出并在运输途中，目前无法再人工加快配送速度。建议您留意物流更新，如有异常我可以继续帮您核实。",
        "您好，理解您着急收货的心情。由于包裹已经在运输途中，我们暂时无法人工优先配送，建议您先查看物流轨迹，如有异常我们会协助跟进。",
    ]
    rule = "订单运输中无法人工加急，只能持续关注物流轨迹，不得承诺已开启紧急处理。"
    idx = start_idx
    for question, product_category, emotion, output in product(
        questions, PRODUCT_CATEGORIES[:4], ["neutral", "anxious", "angry"], outputs
    ):
        records.append(
            make_record(
                idx,
                "物流",
                "催发货",
                "medium",
                question,
                product_category,
                "运输中",
                emotion,
                rule,
                output,
                ["催发货", "运输中不可加急", "no_overpromise_logistics"],
            )
        )
        idx += 1
    return idx


def build_address_change(records: list[dict], start_idx: int) -> int:
    shipped_questions = [
        "发货后还能改地址吗？",
        "运输中还能改收货地址吗？",
        "包裹已经寄出了还能换地址吗？",
    ]
    shipped_outputs = [
        "您好，订单发货后通常无法直接在后台修改地址。建议您先联系快递员或快递公司尝试协商转寄，具体结果需以物流方反馈为准。",
        "您好，您的包裹已经在配送途中，目前无法直接修改地址。建议您联系派件员协商转寄，如有后续异常我们也可以继续协助跟进。",
    ]
    shipped_rule = "订单发货后通常无法直接修改地址，可建议联系派件员或快递协商，不得承诺已联系仓库改址。"

    pending_questions = [
        "能修改收货地址吗？",
        "订单还没发能帮我改地址吗？",
        "收货地址填错了还能改吗？",
    ]
    pending_outputs = [
        "您好，当前订单还未发货，您可以先尝试在订单页面修改收货地址。如果页面暂时无法操作，我们也可以帮您进一步核实，但最终仍需以仓库处理结果为准。",
        "您好，订单目前还是待发货状态，可以尝试修改地址。不过是否修改成功还要以系统和仓库拦截结果为准，建议您尽快操作。",
    ]
    pending_rule = "订单待发货时可尝试修改地址，但修改结果需以仓库拦截或系统处理结果为准。"

    idx = start_idx
    for question, product_category, emotion, output in product(
        shipped_questions, PRODUCT_CATEGORIES[:4], ["neutral", "anxious"], shipped_outputs
    ):
        records.append(
            make_record(
                idx,
                "物流",
                "地址修改",
                "medium",
                question,
                product_category,
                "运输中",
                emotion,
                shipped_rule,
                output,
                ["地址修改", "发货后不可改地址", "no_overpromise_address"],
            )
        )
        idx += 1

    for question, product_category, emotion, output in product(
        pending_questions, PRODUCT_CATEGORIES[:4], ["neutral", "anxious"], pending_outputs
    ):
        records.append(
            make_record(
                idx,
                "物流",
                "地址修改",
                "easy",
                question,
                product_category,
                "待发货",
                emotion,
                pending_rule,
                output,
                ["地址修改", "待发货可尝试改地址", "no_overpromise_address"],
            )
        )
        idx += 1
    return idx


def build_complaints(records: list[dict], start_idx: int) -> int:
    complaint_specs = [
        (
            "质量太差了怎么办？",
            "待发货",
            "如果您目前对商品质量已有疑虑，且订单还未发货，我们可以协助您拦截订单或申请退款。您也可以把订单号发给我，我帮您进一步核实并跟进处理。",
        ),
        (
            "质量太差了怎么办？",
            "运输中",
            "非常抱歉给您带来了不好的体验。商品目前还在运输途中，建议您收到后先检查。如果确认存在质量问题，您可以第一时间联系我们，我们会协助您处理退换货。",
        ),
        (
            "我要投诉",
            "待发货",
            "非常抱歉给您带来了不好的体验。为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会立即为您反馈并持续跟进。",
        ),
        (
            "商家态度不好怎么办？",
            "已签收",
            "非常抱歉给您带来了不愉快的体验。请您提供订单号和具体沟通记录，我们会尽快为您核实情况并反馈给对应专员跟进处理。",
        ),
    ]
    rule = "投诉场景需先安抚情绪，再给出明确处理路径或升级方式，不得虚构已安排专员、已重发或固定反馈时限。"
    idx = start_idx
    for (question, order_status, output), product_category, emotion in product(
        complaint_specs, ["家居用品", "食品", "母婴用品"], ["anxious", "angry"]
    ):
        records.append(
            make_record(
                idx,
                "投诉",
                "投诉安抚",
                "hard",
                question,
                product_category,
                order_status,
                emotion,
                rule,
                output,
                ["投诉安抚", "先安抚再处理", "no_overpromise_complaint"],
            )
        )
        idx += 1
    return idx


def build_product_unknown(records: list[dict], start_idx: int) -> int:
    specs = [
        (
            "库存还有吗？",
            "您好，商品库存是实时变动的，建议您以当前商品详情页显示的信息为准。如果您需要进一步确认，可以把商品名称或链接发给我，我帮您核实。",
        ),
        (
            "这个商品适合什么人群？",
            "您好，具体适用人群建议您优先参考商品详情页中的商品说明或参数介绍。如果页面信息还不够清楚，您可以把商品链接发给我，我帮您进一步核实。",
        ),
        (
            "支持保修吗？",
            "您好，具体保修时长和范围建议您优先查看商品详情页或售后说明。如果页面信息不够清楚，您可以把商品名称或链接发给我，我帮您进一步核实。",
        ),
    ]
    rule = "当上下文未提供明确商品属性时，应以商品详情页为准或说明可进一步核实，不得编造成分、材质、库存或适用人群。"
    idx = start_idx
    for (question, output), product_category, emotion in product(
        specs, PRODUCT_CATEGORIES, ["neutral", "anxious"]
    ):
        records.append(
            make_record(
                idx,
                "售前",
                "商品咨询",
                "medium",
                question,
                product_category,
                "待发货",
                emotion,
                rule,
                output,
                ["商品咨询", "信息不足先说明", "no_hallucination_product_info"],
            )
        )
        idx += 1
    return idx


def build_quality_refund_exchange(records: list[dict], start_idx: int) -> int:
    specs = [
        (
            "退货退款",
            "这个商品可以退吗？",
            "商品存在质量问题是可以申请退货退款的。请您在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快处理，相关运费由商家承担。",
            ["退货退款", "质量问题可退", "quality_issue_no_unopened_constraint"],
        ),
        (
            "换货补发",
            "可以换货吗？",
            "您好，如果商品存在质量问题，是可以申请换货的。请您提供订单号和商品问题照片，我们核实后会尽快为您安排换货或补发处理。",
            ["换货补发", "质量问题可换货", "quality_issue_no_unopened_constraint"],
        ),
    ]
    idx = start_idx
    for (subcategory, question, output, tags), product_category, emotion in product(
        specs, ["护肤品", "数码配件", "食品", "母婴用品"], ["neutral", "anxious"]
    ):
        platform_rule = (
            "商品存在质量问题时可申请退货退款，并由商家承担相关运费。"
            if subcategory == "退货退款"
            else "发错货或商品损坏时，可优先安排补发或换货。"
        )
        records.append(
            make_record(
                idx,
                "售后",
                subcategory,
                "medium" if subcategory == "退货退款" else "easy",
                question,
                product_category,
                "已签收",
                emotion,
                platform_rule,
                output,
                tags,
                is_opened=True,
                has_quality_issue=True,
            )
        )
        idx += 1
    return idx


def main() -> None:
    records: list[dict] = []
    idx = 1
    idx = build_logistics_in_transit(records, idx)
    idx = build_address_change(records, idx)
    idx = build_complaints(records, idx)
    idx = build_product_unknown(records, idx)
    idx = build_quality_refund_exchange(records, idx)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
