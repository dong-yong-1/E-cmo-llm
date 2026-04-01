from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "ecom_cs_schema_v1"

INPUT_FIELD_ORDER = [
    "product_category",
    "order_status",
    "is_opened",
    "has_quality_issue",
    "user_emotion",
    "platform_rule",
]

FIELD_LABELS = {
    "product_category": "商品类别",
    "order_status": "订单状态",
    "is_opened": "是否拆封",
    "has_quality_issue": "是否质量问题",
    "user_emotion": "用户情绪",
    "platform_rule": "平台规则",
}

BOOL_DISPLAY = {
    True: "是",
    False: "否",
}

REQUIRED_FIELDS = {
    "id",
    "category",
    "subcategory",
    "instruction",
    "input",
    "output",
    "policy_tags",
    "quality_label",
    "difficulty",
    "source",
    "schema_version",
}


def normalize_field_value(value: Any) -> str:
    if isinstance(value, bool):
        return BOOL_DISPLAY[value]
    if value is None:
        return "未知"
    return str(value)


def format_input_context(input_data: Any) -> str:
    if isinstance(input_data, dict):
        lines = []
        for key in INPUT_FIELD_ORDER:
            if key in input_data:
                lines.append(f"{FIELD_LABELS.get(key, key)}：{normalize_field_value(input_data[key])}")
        for key, value in input_data.items():
            if key not in INPUT_FIELD_ORDER:
                lines.append(f"{FIELD_LABELS.get(key, key)}：{normalize_field_value(value)}")
        return "\n".join(lines)
    if input_data is None:
        return ""
    return str(input_data)


def build_prompt(record: dict[str, Any]) -> str:
    prompt = record["instruction"]
    context = format_input_context(record.get("input"))
    if context:
        prompt += f"\n{context}"
    return prompt


def validate_record(record: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise ValueError(f"样本缺少字段: {sorted(missing)}")

    if not isinstance(record["input"], dict):
        raise ValueError("字段 input 必须为 dict。")

    if not isinstance(record["policy_tags"], list) or not record["policy_tags"]:
        raise ValueError("字段 policy_tags 必须是非空列表。")

    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version 不匹配，期望 {SCHEMA_VERSION}，实际为 {record['schema_version']}"
        )
