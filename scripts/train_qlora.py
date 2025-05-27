#!/usr/bin/env python
"""
Train DistilGPT-2 on an Alpaca subset with either
- QLoRA (4-bit + LoRA) **default**
- full-parameter fine-tuning (FP16)

Usage:
    python scripts/train_qlora.py --method qlora   # default
    python scripts/train_qlora.py --method full    # baseline
"""

import argparse
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch


def build_model(method: str, model_name: str):
    """
    Return (model, tokenizer) according to the chosen method.
    method ∈ {"qlora", "full"}
    """
    if method == "qlora":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=bnb_cfg
        )

        lora_cfg = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_cfg)

    elif method == "full":
        # no quantization, no LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
    else:
        raise ValueError("method must be 'qlora' or 'full'")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main(args):
    model, tokenizer = build_model(args.method, args.model_name)

    ds = load_from_disk(args.dataset).train_test_split(test_size=0.1, seed=42)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
    )

    Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
    ).train()

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["qlora", "full"], default="qlora")
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--dataset", default="../data/alpaca_prepared")
    p.add_argument("--out_dir", default="../checkpoints/train_out")
    main(p.parse_args())
