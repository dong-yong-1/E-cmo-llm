from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from dataset_schema import SCHEMA_VERSION, validate_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "targeted" / "precision_targeted_v1.json"


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
        "id": f"precision_{category}_{subcategory}_{idx:05d}",
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
        "source": "synthetic_targeted_precision",
        "schema_version": SCHEMA_VERSION,
    }
    validate_record(record)
    return record


def build_refund_reverse(records: list[dict], start_idx: int) -> int:
    specs = [
        (
            "不满意可以退款吗？",
            "已签收",
            True,
            False,
            "商品已拆封且非质量问题时，通常不支持7天无理由退货。",
            "您好，若商品已经拆封且没有质量问题，通常不支持7天无理由退货。若您希望进一步核实特殊情况，可以把订单号和具体诉求发给我，我帮您继续确认。",
            ["退货退款", "不可无理由退货", "refund_opened_non_quality_precise"],
        ),
        (
            "这个商品可以退吗？",
            "已签收",
            True,
            False,
            "商品已拆封且非质量问题时，通常不支持7天无理由退货。",
            "您好，根据平台规则，商品已拆封且没有质量问题时，通常不支持7天无理由退货。如您担心商品存在异常，也可以把具体情况发给我，我帮您进一步核实。",
            ["退货退款", "不可无理由退货", "refund_opened_non_quality_precise"],
        ),
        (
            "这个商品可以退吗？",
            "已签收",
            False,
            False,
            "商品未拆封且不影响二次销售，支持7天无理由退货。",
            "您好，如果商品未拆封且不影响二次销售，一般支持7天无理由退货。您可以在订单页面提交退货申请，我们会按流程为您处理。",
            ["退货退款", "可无理由退货", "refund_unopened_precise"],
        ),
        (
            "可以换货吗？",
            "已签收",
            True,
            True,
            "发错货或商品损坏时，可优先安排补发或换货，需先提交售后申请并等待审核。",
            "您好，如果商品存在质量问题，是可以申请换货的。请您先在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快为您安排后续处理。",
            ["换货补发", "先提交售后", "quality_exchange_precise"],
        ),
    ]
    idx = start_idx
    for (
        question,
        order_status,
        is_opened,
        has_quality_issue,
        platform_rule,
        output,
        tags,
    ), product_category, emotion in product(
        specs,
        ["护肤品", "食品", "家居用品", "数码配件"],
        ["neutral", "anxious"],
    ):
        subcategory = "换货补发" if "换货" in tags[0] else "退货退款"
        records.append(
            make_record(
                idx,
                "售后",
                subcategory,
                "medium" if subcategory == "退货退款" else "easy",
                question,
                product_category,
                order_status,
                emotion,
                platform_rule,
                output,
                tags,
                is_opened=is_opened,
                has_quality_issue=has_quality_issue,
            )
        )
        idx += 1
    return idx


def build_complaint_no_deadline(records: list[dict], start_idx: int) -> int:
    specs = [
        (
            "我要投诉",
            "待发货",
            "非常抱歉给您带来了不好的体验。为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会立即为您反馈并持续跟进。",
        ),
        (
            "我要投诉",
            "运输中",
            "非常抱歉给您带来了不好的体验。为了尽快帮您核实处理，麻烦您提供一下订单号和具体问题描述，我会先为您记录并继续跟进处理进度。",
        ),
        (
            "质量太差了怎么办？",
            "运输中",
            "非常抱歉给您带来了不好的体验。商品目前还在运输途中，建议您收到后先检查。如果确认存在质量问题，您可以第一时间联系我们，我们会协助您处理退换货。",
        ),
    ]
    rule = "投诉场景需先安抚情绪，再给出明确处理路径；不得承诺24小时反馈、短信电话通知、已安排专员或已重新寄送。"
    idx = start_idx
    for (question, order_status, output), product_category, emotion in product(
        specs,
        ["家居用品", "食品", "母婴用品"],
        ["anxious", "angry"],
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
                ["投诉安抚", "先安抚再处理", "complaint_no_deadline_promise"],
            )
        )
        idx += 1
    return idx


def build_exchange_flow(records: list[dict], start_idx: int) -> int:
    specs = [
        (
            "商品坏了可以补发吗？",
            "您好，如果商品存在破损或质量问题，可以申请补发或换货。请您先在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快为您安排处理。",
        ),
        (
            "可以换货吗？",
            "您好，可以申请换货的。请您先在订单页面提交售后换货申请，审核通过后我们会按流程为您安排后续处理，具体结果以审核为准。",
        ),
    ]
    rule = "换货或补发通常需要用户先在订单页面提交售后申请，审核通过后再安排处理，不得承诺固定联系时限。"
    idx = start_idx
    for (question, output), product_category, emotion in product(
        specs,
        ["母婴用品", "数码配件", "家居用品"],
        ["neutral", "anxious"],
    ):
        records.append(
            make_record(
                idx,
                "售后",
                "换货补发",
                "easy",
                question,
                product_category,
                "已签收",
                emotion,
                rule,
                output,
                ["换货补发", "先提交售后", "exchange_flow_precise"],
                is_opened=True,
                has_quality_issue=True,
            )
        )
        idx += 1
    return idx


def main() -> None:
    records: list[dict] = []
    idx = 1
    idx = build_refund_reverse(records, idx)
    idx = build_complaint_no_deadline(records, idx)
    idx = build_exchange_flow(records, idx)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
