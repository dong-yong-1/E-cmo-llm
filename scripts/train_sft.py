import argparse
import inspect
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from dataset_schema import build_prompt, validate_record

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电商客服 SFT + LoRA 训练脚本")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型名称")
    parser.add_argument("--data-path", default="./ecommerce_sft_v1.json", help="训练数据路径")
    parser.add_argument("--output-dir", default="./models/sft_demo", help="模型输出目录")
    parser.add_argument("--max-length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch-size", type=int, default=2, help="单卡 batch size")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--save-steps", type=int, default=50, help="checkpoint 保存间隔")
    parser.add_argument("--save-total-limit", type=int, default=2, help="最多保留多少个 checkpoint")
    parser.add_argument("--logging-steps", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="warmup 比例")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="LoRA target modules，多个模块用英文逗号分隔",
    )
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="模型加载精度")
    parser.add_argument("--device-map", default="cpu", help="device_map 参数，默认兼容本地 CPU")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="训练后 demo 生成长度")
    parser.add_argument("--demo-samples", type=int, default=3, help="训练后展示多少条样例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-train-samples", type=int, default=0, help="仅用于小实验，>0 时只取前 N 条样本训练")
    parser.add_argument("--report-to", default="none", help="训练日志上报目标，例如 none 或 wandb")
    parser.add_argument("--run-name", default=None, help="实验名称，启用 wandb 时建议设置")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_dataset_records(data_path: str, max_train_samples: int) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    normalized_records = []
    for record in raw_dataset:
        validate_record(record)
        normalized_records.append(record)
        if max_train_samples > 0 and len(normalized_records) >= max_train_samples:
            break
    return normalized_records


def build_train_dataset(records: list[dict]) -> Dataset:
    train_data = [{"prompt": build_prompt(record), "completion": record["output"]} for record in records]
    return Dataset.from_list(train_data)


def load_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, hf_token: str, torch_dtype: torch.dtype, device_map: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    if not target_modules:
        raise ValueError("target_modules 不能为空，请至少传入一个模块名。")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
    )


def print_run_config(args: argparse.Namespace, num_samples: int) -> None:
    print("训练配置：")
    print(f"- model_name: {args.model_name}")
    print(f"- data_path: {args.data_path}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- num_samples: {num_samples}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- epochs: {args.epochs}")
    print(f"- learning_rate: {args.learning_rate}")
    print(f"- lora_r: {args.lora_r}")
    print(f"- lora_alpha: {args.lora_alpha}")
    print(f"- lora_dropout: {args.lora_dropout}")
    print(f"- target_modules: {args.target_modules}")
    print(f"- torch_dtype: {args.torch_dtype}")
    print(f"- device_map: {args.device_map}")
    print(f"- report_to: {args.report_to}")
    print(f"- run_name: {args.run_name}")


def generate_demo(model, tokenizer, query: str, max_new_tokens: int) -> str:
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("缺少环境变量 HF_TOKEN，请先在 .env 中配置。")

    records = load_dataset_records(args.data_path, args.max_train_samples)
    train_dataset = build_train_dataset(records)
    print_run_config(args, len(train_dataset))

    tokenizer = load_tokenizer(args.model_name, hf_token)
    model = load_model(
        args.model_name,
        hf_token,
        get_torch_dtype(args.torch_dtype),
        args.device_map,
    )

    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": build_training_args(args),
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_signature.parameters:
        trainer_kwargs["max_seq_length"] = args.max_length
    else:
        print("提示: 当前 TRL 版本不支持 max_seq_length 参数，将使用默认序列长度策略。")

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n训练完成，生成示例：")
    for sample in train_dataset.select(range(min(args.demo_samples, len(train_dataset)))):
        print("用户:", sample["prompt"])
        print("模型:", generate_demo(model, tokenizer, sample["prompt"], args.max_new_tokens))
        print("-" * 50)


if __name__ == "__main__":
    main()
