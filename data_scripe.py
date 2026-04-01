import argparse
import json
import os
import random
import time
from itertools import count

from dotenv import load_dotenv
from openai import OpenAI

from scripts.dataset_schema import SCHEMA_VERSION, validate_record

load_dotenv()

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

SCENARIOS = [
    {
        "category": "售后",
        "subcategory": "退货退款",
        "question_templates": [
            "这个商品可以退吗？",
            "已经拆封还能退吗？",
            "不满意可以退款吗？",
        ],
        "rules": [
            {
                "tag": "可无理由退货",
                "platform_rule": "商品未拆封且不影响二次销售，支持7天无理由退货。",
                "is_opened": False,
                "has_quality_issue": False,
                "difficulty": "easy",
            },
            {
                "tag": "不可无理由退货",
                "platform_rule": "拆封后非质量问题通常不支持7天无理由退货。",
                "is_opened": True,
                "has_quality_issue": False,
                "difficulty": "medium",
            },
            {
                "tag": "质量问题可退",
                "platform_rule": "商品存在质量问题时可申请退货退款，并由商家承担相关运费。",
                "is_opened": True,
                "has_quality_issue": True,
                "difficulty": "medium",
            },
        ],
        "order_status": ["已签收"],
    },
    {
        "category": "售后",
        "subcategory": "换货补发",
        "question_templates": [
            "发错货了怎么办？",
            "商品坏了可以补发吗？",
            "可以换货吗？",
        ],
        "rules": [
            {
                "tag": "商家补发",
                "platform_rule": "发错货或商品损坏时，可优先安排补发或换货。",
                "is_opened": True,
                "has_quality_issue": True,
                "difficulty": "easy",
            },
            {
                "tag": "先提交售后",
                "platform_rule": "如需换货，请先在订单页面提交售后申请并等待审核。",
                "is_opened": False,
                "has_quality_issue": False,
                "difficulty": "easy",
            },
        ],
        "order_status": ["已签收"],
    },
    {
        "category": "物流",
        "subcategory": "催发货",
        "question_templates": [
            "为什么还没发货？",
            "订单多久发货？",
            "可以加急发货吗？",
        ],
        "rules": [
            {
                "tag": "待发货可催单",
                "platform_rule": "订单待发货时可帮助催仓，但无法保证立即发出。",
                "difficulty": "easy",
            },
            {
                "tag": "运输中不可加急",
                "platform_rule": "商品已发出后无法人工加急，可持续关注物流轨迹。",
                "difficulty": "medium",
            },
        ],
        "order_status": ["待发货", "运输中"],
    },
    {
        "category": "物流",
        "subcategory": "地址修改",
        "question_templates": [
            "能修改收货地址吗？",
            "发货后还能改地址吗？",
        ],
        "rules": [
            {
                "tag": "未发货可改地址",
                "platform_rule": "订单未发货前可尝试修改地址，但需以仓库拦截结果为准。",
                "difficulty": "easy",
            },
            {
                "tag": "发货后不可改地址",
                "platform_rule": "订单发货后通常无法直接修改地址，可联系快递尝试协商。",
                "difficulty": "medium",
            },
        ],
        "order_status": ["待发货", "运输中"],
    },
    {
        "category": "售前",
        "subcategory": "商品咨询",
        "question_templates": [
            "这个商品适合什么人群？",
            "库存还有吗？",
            "支持保修吗？",
            "材质是什么？",
        ],
        "rules": [
            {
                "tag": "正常介绍商品",
                "platform_rule": "商品咨询需准确介绍规格、适用人群与售后范围，不得夸大承诺。",
                "difficulty": "easy",
            },
            {
                "tag": "信息不足先说明",
                "platform_rule": "当商品信息不足时，应先说明以页面展示或进一步核实为准。",
                "difficulty": "medium",
            },
        ],
        "order_status": ["待发货"],
    },
    {
        "category": "投诉",
        "subcategory": "投诉安抚",
        "question_templates": [
            "客服一直不回复",
            "我要投诉",
            "商家态度不好怎么办？",
            "质量太差了怎么办？",
        ],
        "rules": [
            {
                "tag": "先安抚再处理",
                "platform_rule": "投诉场景需优先安抚情绪，再给出明确处理路径或升级方式。",
                "difficulty": "hard",
            },
            {
                "tag": "不可空泛道歉",
                "platform_rule": "不能只道歉不处理，需要给出具体补救或转人工方案。",
                "difficulty": "hard",
            },
        ],
        "order_status": ["已签收", "运输中", "待发货"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成符合 schema v1 的电商客服数据。")
    parser.add_argument("--output", default="ecommerce_sft_v1.json", help="输出文件路径")
    parser.add_argument("--num-samples", type=int, default=3000, help="目标样本数量")
    parser.add_argument("--model", default="deepseek-chat", help="用于生成回复的模型名")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="每次请求后休眠秒数")
    return parser.parse_args()


def build_record(index: int, scenario: dict, rule: dict, rng: random.Random) -> dict:
    product_category = rng.choice(PRODUCT_CATEGORIES)
    order_status = rng.choice(scenario["order_status"])
    user_emotion = rng.choice(USER_EMOTIONS)
    question = rng.choice(scenario["question_templates"])

    input_data = {
        "product_category": product_category,
        "order_status": order_status,
        "is_opened": rule.get("is_opened", order_status == "已签收"),
        "has_quality_issue": rule.get("has_quality_issue", False),
        "user_emotion": user_emotion,
        "platform_rule": rule["platform_rule"],
    }

    slug = f"{scenario['category']}_{scenario['subcategory']}_{index:05d}".replace(" ", "_")
    return {
        "id": slug,
        "category": scenario["category"],
        "subcategory": scenario["subcategory"],
        "instruction": f"用户问：{question}",
        "input": input_data,
        "output": "",
        "policy_tags": [scenario["subcategory"], rule["tag"]],
        "quality_label": "good",
        "difficulty": rule["difficulty"],
        "source": "synthetic",
        "schema_version": SCHEMA_VERSION,
    }


def build_generation_prompt(record: dict) -> str:
    input_data = record["input"]
    return f"""
你是专业电商客服数据生成助手。请基于给定场景，生成 1 条高质量中文客服回复。

要求：
1. 回复长度控制在 45 到 90 字。
2. 语气礼貌、自然，适合真实电商客服。
3. 必须严格遵守平台规则，不能乱承诺。
4. 投诉场景先安抚，再给解决方案。
5. 如果信息不足，可以说明会帮助核实，但不要编造不存在的权益。
6. 只输出客服回复正文，不要加编号、引号或解释。

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

    scenario_cycle = count()
    while len(dataset) < args.num_samples:
        scenario = SCENARIOS[next(scenario_cycle) % len(SCENARIOS)]
        rule = rng.choice(scenario["rules"])
        record = build_record(len(dataset) + 1, scenario, rule, rng)

        try:
            record["output"] = generate_output(record, args.model)
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
