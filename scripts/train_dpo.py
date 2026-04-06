from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电商客服小规模 DPO 训练脚本")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型名称")
    parser.add_argument("--sft-adapter-path", default="./models/final_v4_clean_sft_r16_qkvo", help="SFT adapter 路径")
    parser.add_argument("--data-path", default="./data/dpo/small_dpo_pairs_v1.json", help="DPO 数据路径")
    parser.add_argument("--output-dir", default="./models/small_dpo_v1", help="DPO 模型输出目录")
    parser.add_argument("--max-length", type=int, default=768, help="最大序列长度")
    parser.add_argument("--batch-size", type=int, default=2, help="单卡 batch size")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--save-steps", type=int, default=50, help="checkpoint 保存间隔")
    parser.add_argument("--save-total-limit", type=int, default=2, help="最多保留多少个 checkpoint")
    parser.add_argument("--logging-steps", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="warmup 比例")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="模型加载精度")
    parser.add_argument("--device-map", default="cpu", help="device_map 参数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-train-samples", type=int, default=0, help="仅用于小实验，>0 时只取前 N 条样本")
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


def get_fp_flags(dtype_name: str) -> tuple[bool, bool]:
    return dtype_name == "fp16" or dtype_name == "float16", dtype_name == "bfloat16"


def load_preference_records(data_path: str, max_train_samples: int) -> list[dict]:
    raw = json.loads(Path(data_path).read_text())
    records = []
    required = {"prompt", "chosen", "rejected"}
    for item in raw:
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"DPO 样本缺少字段: {sorted(missing)}")
        records.append(item)
        if max_train_samples > 0 and len(records) >= max_train_samples:
            break
    return records


def build_train_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            }
            for record in records
        ]
    )


def load_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, hf_token: str, torch_dtype: torch.dtype, device_map: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )


def build_dpo_config(args: argparse.Namespace) -> DPOConfig:
    fp16, bf16 = get_fp_flags(args.torch_dtype)
    return DPOConfig(
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
        max_length=args.max_length,
        beta=args.beta,
        fp16=fp16,
        bf16=bf16,
    )


def print_run_config(args: argparse.Namespace, num_samples: int) -> None:
    print("DPO 训练配置：")
    print(f"- model_name: {args.model_name}")
    print(f"- sft_adapter_path: {args.sft_adapter_path}")
    print(f"- data_path: {args.data_path}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- num_samples: {num_samples}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- epochs: {args.epochs}")
    print(f"- learning_rate: {args.learning_rate}")
    print(f"- beta: {args.beta}")
    print(f"- max_length: {args.max_length}")
    print(f"- torch_dtype: {args.torch_dtype}")
    print(f"- device_map: {args.device_map}")
    print(f"- report_to: {args.report_to}")
    print(f"- run_name: {args.run_name}")


def main() -> None:
    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("缺少环境变量 HF_TOKEN，请先在 .env 中配置。")

    records = load_preference_records(args.data_path, args.max_train_samples)
    train_dataset = build_train_dataset(records)
    print_run_config(args, len(train_dataset))

    tokenizer = load_tokenizer(args.model_name, hf_token)
    torch_dtype = get_torch_dtype(args.torch_dtype)

    policy_base = load_base_model(args.model_name, hf_token, torch_dtype, args.device_map)
    model = PeftModel.from_pretrained(policy_base, args.sft_adapter_path, is_trainable=True)

    ref_base = load_base_model(args.model_name, hf_token, torch_dtype, args.device_map)
    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter_path, is_trainable=False)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=build_dpo_config(args),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO 训练完成，模型输出到: {args.output_dir}")


if __name__ == "__main__":
    main()
