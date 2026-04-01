import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_schema import build_prompt, validate_record

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电商客服模型评估脚本：输出规则准确率、胜率与 badcase。")
    parser.add_argument("--eval-data-path", required=True, help="评测集路径，推荐使用 schema v1 JSON")
    parser.add_argument("--candidate-model-name", required=True, help="候选模型基础模型名")
    parser.add_argument("--candidate-adapter-path", default=None, help="候选模型 LoRA adapter 路径")
    parser.add_argument("--candidate-name", default="candidate", help="候选模型展示名")
    parser.add_argument("--baseline-model-name", default=None, help="baseline 基础模型名，默认复用 candidate-model-name")
    parser.add_argument("--baseline-adapter-path", default=None, help="baseline 的 LoRA adapter 路径；不传则评估基座模型")
    parser.add_argument("--baseline-name", default="baseline", help="baseline 展示名")
    parser.add_argument("--judge-model", default="deepseek-chat", help="用于规则判分和胜率比较的 judge 模型")
    parser.add_argument("--output-report", default="reports/eval_report.json", help="评估报告输出路径")
    parser.add_argument("--output-details", default="reports/eval_details.json", help="逐样本明细输出路径")
    parser.add_argument("--max-samples", type=int, default=100, help="最多评测多少条样本")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="生成回复最大长度")
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="本地模型加载精度")
    parser.add_argument("--device-map", default="cpu", help="device_map 参数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip-pairwise", action="store_true", help="仅评估候选模型，不做与 baseline 的胜率比较")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_records(path: str, max_samples: int) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    records = []
    for record in raw_records:
        validate_record(record)
        records.append(record)
        if len(records) >= max_samples:
            break
    return records


def load_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_generation_model(
    model_name: str,
    hf_token: str,
    torch_dtype: torch.dtype,
    device_map: str,
    adapter_path: str | None = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def build_openai_client() -> OpenAI:
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise ValueError("缺少 JUDGE_API_KEY 或 DEEPSEEK_API_KEY，无法调用 judge 模型。")
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"judge 输出中未找到 JSON: {text}")
    return json.loads(text[start:end + 1])


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def normalize_rule_eval(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_correct": to_bool(result.get("rule_correct", False)),
        "reference_match": to_bool(result.get("reference_match", False)),
        "helpfulness_score": int(result.get("helpfulness_score", 1)),
        "politeness_score": int(result.get("politeness_score", 1)),
        "overall_score": int(result.get("overall_score", 1)),
        "error_type": str(result.get("error_type", "其他")),
        "reason": str(result.get("reason", "")),
    }


def normalize_pairwise_result(result: dict[str, Any]) -> dict[str, Any]:
    winner = str(result.get("winner", "Tie")).strip()
    winner_upper = winner.upper()
    if winner_upper == "A":
        normalized_winner = "A"
    elif winner_upper == "B":
        normalized_winner = "B"
    else:
        normalized_winner = "Tie"
    return {
        "winner": normalized_winner,
        "reason": str(result.get("reason", "")),
    }


def judge_rule_following(client: OpenAI, judge_model: str, record: dict[str, Any], response_text: str) -> dict[str, Any]:
    prompt = f"""
你是电商客服评测员。请根据给定样本判断“模型回复是否符合平台规则、是否可直接用于客服场景”。

请只输出 JSON 对象，字段必须完整：
{{
  "rule_correct": true,
  "reference_match": true,
  "helpfulness_score": 1,
  "politeness_score": 1,
  "overall_score": 1,
  "error_type": "none",
  "reason": "简短中文解释"
}}

评分要求：
1. rule_correct: 是否遵守平台规则与 policy_tags，不得乱承诺。
2. reference_match: 是否和参考答案表达的处理方向基本一致。
3. helpfulness_score: 1 到 5。
4. politeness_score: 1 到 5。
5. overall_score: 1 到 5。
6. error_type: 只能从以下值中选择：
   - none
   - 规则错判
   - 过度承诺
   - 信息缺失乱答
   - 安抚不足
   - 回复冗长
   - 其他

样本信息：
- 用户问题：{record["instruction"]}
- 输入上下文：{json.dumps(record["input"], ensure_ascii=False)}
- 平台规则：{record["input"]["platform_rule"]}
- policy_tags：{json.dumps(record["policy_tags"], ensure_ascii=False)}
- 参考答案：{record["output"]}
- 模型回复：{response_text}
""".strip()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return normalize_rule_eval(extract_json_object(response.choices[0].message.content))


def judge_pairwise(
    client: OpenAI,
    judge_model: str,
    record: dict[str, Any],
    candidate_response: str,
    baseline_response: str,
) -> dict[str, Any]:
    prompt = f"""
你是电商客服模型对比评测员。请比较 A 和 B 哪个更适合作为客服最终回复。

请只输出 JSON：
{{
  "winner": "A",
  "reason": "简短中文解释"
}}

winner 只能取值：
- "A"
- "B"
- "Tie"

评判标准优先级：
1. 是否遵守平台规则，不能乱承诺
2. 是否真正解决用户问题
3. 是否礼貌、自然
4. 是否表达简洁

样本信息：
- 用户问题：{record["instruction"]}
- 输入上下文：{json.dumps(record["input"], ensure_ascii=False)}
- 平台规则：{record["input"]["platform_rule"]}
- policy_tags：{json.dumps(record["policy_tags"], ensure_ascii=False)}

A 回复：
{candidate_response}

B 回复：
{baseline_response}
""".strip()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return normalize_pairwise_result(extract_json_object(response.choices[0].message.content))


def aggregate_category_accuracy(details: list[dict[str, Any]], key: str) -> dict[str, Any]:
    bucket = defaultdict(lambda: {"total": 0, "correct": 0})
    for item in details:
        value = item.get(key, "unknown")
        bucket[value]["total"] += 1
        bucket[value]["correct"] += int(item["candidate_eval"]["rule_correct"])

    return {
        name: {
            "total": stat["total"],
            "rule_accuracy": round(stat["correct"] / stat["total"], 4) if stat["total"] else 0.0,
        }
        for name, stat in bucket.items()
    }


def build_report(
    args: argparse.Namespace,
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(details)
    rule_correct = sum(int(item["candidate_eval"]["rule_correct"]) for item in details)
    reference_match = sum(int(item["candidate_eval"]["reference_match"]) for item in details)
    avg_helpfulness = sum(item["candidate_eval"]["helpfulness_score"] for item in details) / total if total else 0.0
    avg_politeness = sum(item["candidate_eval"]["politeness_score"] for item in details) / total if total else 0.0
    avg_overall = sum(item["candidate_eval"]["overall_score"] for item in details) / total if total else 0.0

    pairwise_counts = Counter(item["pairwise"]["winner"] for item in details if item.get("pairwise"))
    wins = pairwise_counts.get("A", 0)
    losses = pairwise_counts.get("B", 0)
    ties = pairwise_counts.get("Tie", 0)
    win_rate = (wins + 0.5 * ties) / total if total else 0.0

    return {
        "meta": {
            "eval_data_path": args.eval_data_path,
            "candidate_name": args.candidate_name,
            "candidate_model_name": args.candidate_model_name,
            "candidate_adapter_path": args.candidate_adapter_path,
            "baseline_name": args.baseline_name,
            "baseline_model_name": args.baseline_model_name or args.candidate_model_name,
            "baseline_adapter_path": args.baseline_adapter_path,
            "judge_model": args.judge_model,
            "num_samples": total,
            "skip_pairwise": args.skip_pairwise,
        },
        "metrics": {
            "rule_accuracy": round(rule_correct / total, 4) if total else 0.0,
            "reference_match_rate": round(reference_match / total, 4) if total else 0.0,
            "avg_helpfulness_score": round(avg_helpfulness, 4),
            "avg_politeness_score": round(avg_politeness, 4),
            "avg_overall_score": round(avg_overall, 4),
            "win_rate_vs_baseline": None if args.skip_pairwise else (round(win_rate, 4) if total else None),
            "pairwise_counts": dict(pairwise_counts),
        },
        "breakdown": {
            "by_category": aggregate_category_accuracy(details, "category"),
            "by_subcategory": aggregate_category_accuracy(details, "subcategory"),
            "by_difficulty": aggregate_category_accuracy(details, "difficulty"),
            "error_type_distribution": dict(Counter(item["candidate_eval"]["error_type"] for item in details)),
        },
        "badcases": [
            {
                "id": item["id"],
                "category": item["category"],
                "subcategory": item["subcategory"],
                "difficulty": item["difficulty"],
                "instruction": item["instruction"],
                "candidate_response": item["candidate_response"],
                "reference_output": item["reference_output"],
                "error_type": item["candidate_eval"]["error_type"],
                "reason": item["candidate_eval"]["reason"],
            }
            for item in details
            if not item["candidate_eval"]["rule_correct"]
        ],
        "summary": {
            "rule_correct_cases": rule_correct,
            "rule_wrong_cases": total - rule_correct,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("缺少环境变量 HF_TOKEN，请先在 .env 中配置。")

    records = load_records(args.eval_data_path, args.max_samples)
    tokenizer = load_tokenizer(args.candidate_model_name, hf_token)

    candidate_model = load_generation_model(
        model_name=args.candidate_model_name,
        hf_token=hf_token,
        torch_dtype=get_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        adapter_path=args.candidate_adapter_path,
    )

    baseline_model_name = args.baseline_model_name or args.candidate_model_name
    baseline_tokenizer = None
    baseline_model = None
    if not args.skip_pairwise:
        baseline_tokenizer = tokenizer if baseline_model_name == args.candidate_model_name else load_tokenizer(baseline_model_name, hf_token)
        baseline_model = load_generation_model(
            model_name=baseline_model_name,
            hf_token=hf_token,
            torch_dtype=get_torch_dtype(args.torch_dtype),
            device_map=args.device_map,
            adapter_path=args.baseline_adapter_path,
        )

    judge_client = build_openai_client()
    details = []

    for idx, record in enumerate(records, start=1):
        prompt = build_prompt(record)
        candidate_response = generate_response(candidate_model, tokenizer, prompt, args.max_new_tokens)
        baseline_response = ""
        pairwise = None
        if not args.skip_pairwise:
            baseline_response = generate_response(baseline_model, baseline_tokenizer, prompt, args.max_new_tokens)
            pairwise = judge_pairwise(
                judge_client,
                args.judge_model,
                record,
                candidate_response,
                baseline_response,
            )

        candidate_eval = judge_rule_following(judge_client, args.judge_model, record, candidate_response)

        details.append(
            {
                "id": record["id"],
                "category": record["category"],
                "subcategory": record["subcategory"],
                "difficulty": record["difficulty"],
                "instruction": record["instruction"],
                "policy_tags": record["policy_tags"],
                "reference_output": record["output"],
                "candidate_response": candidate_response,
                "baseline_response": baseline_response,
                "candidate_eval": candidate_eval,
                "pairwise": pairwise,
            }
        )

        if idx % 10 == 0 or idx == len(records):
            print(f"已完成评测 {idx}/{len(records)}")

    report = build_report(args, details)

    ensure_parent_dir(args.output_report)
    ensure_parent_dir(args.output_details)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.output_details, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print("评测完成。")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"报告已保存到: {args.output_report}")
    print(f"明细已保存到: {args.output_details}")


if __name__ == "__main__":
    main()
