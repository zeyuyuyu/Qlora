
#!/usr/bin/env python
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch, argparse, pathlib, os

def main(args):
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_cfg, device_map="auto")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tok.pad_token = tok.eos_token
    lora_cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    ds = load_from_disk(args.dataset).train_test_split(test_size=0.1, seed=42)
    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(output_dir=args.out_dir, per_device_train_batch_size=4, per_device_eval_batch_size=4,
                              gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=2e-4,
                              fp16=True, logging_steps=20, save_strategy="epoch", evaluation_strategy="epoch", report_to="none")

    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["test"], data_collator=data_collator)
    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--dataset", default="../data/alpaca_prepared")
    p.add_argument("--out_dir", default="../checkpoints/qlora_distilgpt2")
    main(p.parse_args())
