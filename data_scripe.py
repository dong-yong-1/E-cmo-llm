import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from scripts.dataset_schema import SCHEMA_VERSION, validate_record

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_RULES_PATH = PROJECT_ROOT / "configs" / "high_value_rules.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not DEEPSEEK_API_KEY:
    raise ValueError("缺少环境变量 DEEPSEEK_API_KEY，请先在 .env 中配置。")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

PRODUCT_CATEGORIES = ["服饰", "护肤品", "数码配件", "家居用品", "食品", "母婴用品"]
ORDER_STATUS = ["待发货", "运输中", "已签收"]
USER_EMOTIONS = ["neutral", "anxious", "angry"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成符合 schema v1 的电商客服数据。")
    parser.add_argument("--output", default="ecommerce_sft_v1.json", help="输出文件路径")
    parser.add_argument("--num-samples", type=int, default=3000, help="目标样本数量")
    parser.add_argument("--model", default="deepseek-chat", help="用于生成回复的模型名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="每次请求后休眠秒数")
    parser.add_argument("--rules-path", default=str(DEFAULT_RULES_PATH), help="高价值规则 JSON 路径")
    parser.add_argument("--rule-ids", default="", help="定向生成时使用的 rule_id，多个用逗号分隔")
    parser.add_argument("--samples-per-rule", type=int, default=0, help="定向生成时每条规则生成多少条样本")
    parser.add_argument("--source-label", default="synthetic", help="写入数据的 source 字段")
    return parser.parse_args()


def load_rules(rules_path: str) -> list[dict]:
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_rules(all_rules: list[dict], rule_ids: str) -> list[dict]:
    if not rule_ids.strip():
        return all_rules

    selected_ids = [rule_id.strip() for rule_id in rule_ids.split(",") if rule_id.strip()]
    selected_rules = [rule for rule in all_rules if rule["rule_id"] in selected_ids]
    missing = sorted(set(selected_ids) - {rule["rule_id"] for rule in selected_rules})
    if missing:
        raise ValueError(f"未找到这些 rule_id: {missing}")
    return selected_rules


def build_rule_schedule(selected_rules: list[dict], num_samples: int, samples_per_rule: int) -> list[dict]:
    if samples_per_rule > 0:
        schedule = []
        for rule in selected_rules:
            schedule.extend([rule] * samples_per_rule)
        return schedule

    schedule = []
    while len(schedule) < num_samples:
        schedule.extend(selected_rules)
    return schedule[:num_samples]


def build_record(index: int, rule: dict, rng: random.Random, source_label: str) -> dict:
    product_category = rng.choice(rule.get("product_categories", PRODUCT_CATEGORIES))
    order_status = rng.choice(rule.get("order_status_options", ORDER_STATUS))
    user_emotion = rng.choice(rule.get("user_emotions", USER_EMOTIONS))
    question = rng.choice(rule["question_templates"])

    input_data = {
        "product_category": product_category,
        "order_status": order_status,
        "is_opened": rule.get("is_opened", order_status == "已签收"),
        "has_quality_issue": rule.get("has_quality_issue", False),
        "user_emotion": user_emotion,
        "platform_rule": rule["platform_rule"],
    }

    slug = f"{rule['category']}_{rule['subcategory']}_{index:05d}".replace(" ", "_")
    return {
        "id": slug,
        "category": rule["category"],
        "subcategory": rule["subcategory"],
        "instruction": f"用户问：{question}",
        "input": input_data,
        "output": "",
        "policy_tags": [rule["subcategory"], rule["tag"], rule["rule_id"]],
        "quality_label": "good",
        "difficulty": rule["difficulty"],
        "source": source_label,
        "schema_version": SCHEMA_VERSION,
    }


def build_generation_prompt(record: dict) -> str:
    input_data = record["input"]
    allowed = record.get("_allowed", [])
    forbidden = record.get("_forbidden", [])
    condition = record.get("_condition", "")
    return f"""
你是专业电商客服数据生成助手。请基于给定场景，生成 1 条高质量中文客服回复。

要求：
1. 回复长度控制在 45 到 90 字。
2. 语气礼貌、自然，适合真实电商客服。
3. 必须严格遵守平台规则，不能乱承诺。
4. 投诉场景先安抚，再给解决方案。
5. 如果信息不足，可以说明会帮助核实，但不要编造不存在的权益。
6. 只输出客服回复正文，不要加编号、引号或解释。
7. 回复必须满足“允许行为”，绝不能触发“禁止行为”。

场景信息：
- 一级分类：{record["category"]}
- 二级分类：{record["subcategory"]}
- 用户问题：{record["instruction"]}
- 商品类别：{input_data["product_category"]}
- 订单状态：{input_data["order_status"]}
- 是否拆封：{"是" if input_data["is_opened"] else "否"}
- 是否质量问题：{"是" if input_data["has_quality_issue"] else "否"}
- 用户情绪：{input_data["user_emotion"]}
- 平台规则：{input_data["platform_rule"]}
- 规则标签：{", ".join(record["policy_tags"])}
- 难度：{record["difficulty"]}
- 适用条件：{condition}
- 允许行为：{"；".join(allowed) if allowed else "无"}
- 禁止行为：{"；".join(forbidden) if forbidden else "无"}
""".strip()


def generate_output(record: dict, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": build_generation_prompt(record)}],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset = []
    all_rules = load_rules(args.rules_path)
    selected_rules = select_rules(all_rules, args.rule_ids)
    rule_schedule = build_rule_schedule(selected_rules, args.num_samples, args.samples_per_rule)

    for index, rule in enumerate(rule_schedule, start=1):
        record = build_record(index, rule, rng, args.source_label)
        record["_condition"] = rule.get("condition", "")
        record["_allowed"] = rule.get("allowed", [])
        record["_forbidden"] = rule.get("forbidden", [])

        try:
            record["output"] = generate_output(record, args.model)
            record.pop("_condition", None)
            record.pop("_allowed", None)
            record.pop("_forbidden", None)
            validate_record(record)
            dataset.append(record)
            if len(dataset) % 100 == 0:
                print(f"已生成 {len(dataset)} 条样本")
            time.sleep(args.sleep_seconds)
        except Exception as exc:
            print(f"生成失败，样本 {record['id']}，错误: {exc}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"生成完成，共 {len(dataset)} 条数据，输出到 {args.output}")


if __name__ == "__main__":
    main()
