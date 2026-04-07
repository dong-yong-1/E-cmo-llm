from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from dataset_schema import SCHEMA_VERSION, validate_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "targeted" / "minimal_pair_targeted_v1.json"

PRODUCT_CATEGORIES = ["护肤品", "食品", "母婴用品", "家居用品", "数码配件", "服饰"]
USER_EMOTIONS = ["neutral", "anxious"]
QUESTIONS = [
    "这个商品可以退吗？",
    "不满意可以退款吗？",
    "可以退货吗？",
    "能退款吗？",
]


def make_record(
    idx: int,
    question: str,
    product_category: str,
    user_emotion: str,
    is_opened: bool,
    has_quality_issue: bool,
    platform_rule: str,
    output: str,
    policy_tags: list[str],
) -> dict:
    record = {
        "id": f"minimal_pair_售后_退货退款_{idx:05d}",
        "category": "售后",
        "subcategory": "退货退款",
        "instruction": f"用户问：{question}",
        "input": {
            "product_category": product_category,
            "order_status": "已签收",
            "is_opened": is_opened,
            "has_quality_issue": has_quality_issue,
            "user_emotion": user_emotion,
            "platform_rule": platform_rule,
        },
        "output": output,
        "policy_tags": policy_tags,
        "quality_label": "good",
        "difficulty": "medium",
        "source": "synthetic_targeted_minimal_pair",
        "schema_version": SCHEMA_VERSION,
    }
    validate_record(record)
    return record


def build_records() -> list[dict]:
    records: list[dict] = []
    idx = 1

    for question, product_category, user_emotion in product(QUESTIONS, PRODUCT_CATEGORIES, USER_EMOTIONS):
        records.append(
            make_record(
                idx,
                question,
                product_category,
                user_emotion,
                False,
                False,
                "商品未拆封且不影响二次销售，支持7天无理由退货。",
                (
                    f"您好，根据平台规则，{product_category}在未拆封且不影响二次销售的情况下，"
                    "支持7天无理由退货。您可以在订单页面提交退货申请，我们会按流程为您处理。"
                ),
                ["退货退款", "可无理由退货", "minimal_pair_unopened_refund"],
            )
        )
        idx += 1

        records.append(
            make_record(
                idx,
                question,
                product_category,
                user_emotion,
                True,
                False,
                "商品已拆封且非质量问题时，通常不支持7天无理由退货。",
                (
                    f"您好，根据平台规则，{product_category}在已拆封且无质量问题的情况下，"
                    "通常不支持7天无理由退货。如果您有其他特殊情况，可以把订单号和具体诉求发给我，我帮您进一步核实。"
                ),
                ["退货退款", "不可无理由退货", "minimal_pair_opened_non_quality_refund"],
            )
        )
        idx += 1

        records.append(
            make_record(
                idx,
                question,
                product_category,
                user_emotion,
                True,
                True,
                "商品存在质量问题时可申请退货退款，并由商家承担相关运费。",
                (
                    f"您好，根据平台规则，如果{product_category}存在质量问题，即使已经拆封，"
                    "也是可以申请退货退款的。请您在订单页面提交售后申请并上传相关凭证，我们核实后会按流程处理，相关运费由商家承担。"
                ),
                ["退货退款", "质量问题可退", "minimal_pair_quality_issue_refund"],
            )
        )
        idx += 1

    return records


def main() -> None:
    records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
