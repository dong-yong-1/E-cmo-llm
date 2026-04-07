from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from dataset_schema import build_prompt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "dpo" / "small_dpo_pairs_v2.json"


def make_prompt(question: str, input_data: dict) -> str:
    return build_prompt({"instruction": f"用户问：{question}", "input": input_data})


def make_example(
    idx: int,
    category: str,
    subcategory: str,
    difficulty: str,
    question: str,
    input_data: dict,
    chosen: str,
    rejected: str,
    preference_type: str,
) -> dict:
    return {
        "id": f"dpo_v2_{category}_{subcategory}_{idx:05d}",
        "category": category,
        "subcategory": subcategory,
        "difficulty": difficulty,
        "preference_type": preference_type,
        "prompt": make_prompt(question, input_data),
        "chosen": chosen,
        "rejected": rejected,
    }


def build_shipping_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx

    transit_specs = [
        (
            "可以加急发货吗？",
            "您好，理解您着急收货的心情。您的订单已经在运输途中，目前无法再人工加急。建议您先查看订单页面的物流轨迹，如有异常我们也会继续协助跟进。",
            "您好，您的订单已经在运输途中，目前暂时无法再人工加急。我们这边也会继续关注进度，您稍后可以留意一下物流更新。",
        ),
        (
            "为什么还没发货？",
            "您好，您的订单目前显示为运输中，说明商品已经从仓库发出，正在配送途中。您可以在订单页面查看最新物流轨迹，如有异常我们也会继续帮您跟进。",
            "您好，您的订单已经在运输途中，目前暂时无法再催发。建议您先留意订单页面的物流更新，我们也会继续关注后续进度。",
        ),
        (
            "订单多久发货？",
            "您好，您的订单已从仓库发出，目前正在运输途中。物流信息会持续更新，您可以在订单页面查看实时轨迹，我们也会继续关注配送进度。",
            "您好，您的订单已经发出，目前无法再人工干预提速。建议您留意订单页面的物流更新，如有异常我们再帮您继续核实。",
        ),
    ]

    pending_specs = [
        (
            "可以加急发货吗？",
            "您好，理解您想尽快收到商品的心情。当前订单还是待发货状态，我们可以帮您催仓，但暂时无法保证具体发出时间，您可以留意订单页面的后续更新。",
            "您好，目前订单还是待发货状态，我们这边可以帮您催仓处理。发货时间还要以仓库实际安排为准，辛苦您再留意一下订单更新。",
        ),
        (
            "为什么还没发货？",
            "您好，当前订单还在待发货状态，我们可以帮您催仓跟进，但暂时无法保证具体发出时间。发货后物流信息会及时更新，辛苦您留意一下。",
            "您好，您的订单目前还是待发货状态，我们会继续帮您催仓处理。具体发货时间还要以仓库实际安排为准，您可以先关注订单页面。",
        ),
        (
            "订单多久发货？",
            "您好，您的订单目前处于待发货状态。通常发货后物流信息会更新，您也可以持续关注订单页面；我们这边可以帮您催仓，但暂时无法保证具体时效。",
            "您好，您的订单目前还是待发货状态，我们这边可以帮您催仓。具体多久发出还要以仓库处理进度为准，建议您先留意一下订单页面。",
        ),
    ]

    for question, chosen, rejected in transit_specs:
        for product_category, emotion in product(["食品", "护肤品", "母婴用品"], ["neutral", "anxious"]):
            input_data = {
                "product_category": product_category,
                "order_status": "运输中",
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "订单运输中无法人工加急，只能持续关注物流轨迹，不得承诺已开启紧急处理。",
            }
            records.append(
                make_example(
                    idx, "物流", "催发货", "medium", question, input_data, chosen, rejected, "more_grounded_vs_less_grounded"
                )
            )
            idx += 1

    for question, chosen, rejected in pending_specs:
        for product_category, emotion in product(["服饰", "数码配件", "家居用品"], ["neutral", "angry"]):
            input_data = {
                "product_category": product_category,
                "order_status": "待发货",
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "订单待发货时可帮助催仓，但无法保证立即发出。",
            }
            records.append(
                make_example(
                    idx, "物流", "催发货", "medium", question, input_data, chosen, rejected, "natural_safe_vs_slightly_vague"
                )
            )
            idx += 1

    return idx


def build_complaint_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx
    specs = [
        (
            "客服一直不回复",
            "非常抱歉给您带来了不好的体验。为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会先为您记录并继续跟进。",
            "非常抱歉给您带来了不好的体验。您先把问题和订单号发给我，我这边先帮您登记一下，后续我们也会继续关注处理进度。",
        ),
        (
            "我要投诉",
            "非常抱歉给您带来了不好的体验。为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会先为您记录并继续跟进。",
            "非常理解您现在的不满。您可以先把订单号和具体问题发给我，我这边先记录下来，再帮您继续跟进处理。",
        ),
        (
            "质量太差了怎么办？",
            "非常抱歉给您带来了不好的体验。商品目前还在运输途中，建议您收到后先检查。如果确认存在质量问题，您可以第一时间联系我们，我们会协助您处理退换货。",
            "非常抱歉给您带来了不好的体验。商品目前还在运输途中，您收到后可以先检查一下，如果确实存在问题我们再继续帮您跟进处理。",
        ),
    ]

    for question, chosen, rejected in specs:
        for order_status, product_category, emotion in product(
            ["待发货", "运输中"], ["食品", "家居用品", "母婴用品"], ["anxious", "angry"]
        ):
            input_data = {
                "product_category": product_category,
                "order_status": order_status,
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "投诉场景需先安抚情绪，再给出明确处理路径；不得承诺具体反馈时限、电话短信通知或已安排专员。",
            }
            records.append(
                make_example(
                    idx, "投诉", "投诉安抚", "hard", question, input_data, chosen, rejected, "empathetic_actionable_vs_soft_vague"
                )
            )
            idx += 1

    return idx


def main() -> None:
    records: list[dict] = []
    idx = 1
    idx = build_shipping_pairs(records, idx)
    idx = build_complaint_pairs(records, idx)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
