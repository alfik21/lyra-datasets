#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trening LoRA na Kaggle (GPU P100/T4)
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
import os
import urllib.request
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("Trening LoRA - Lyra Identity Adapter (Kaggle)")
print("=" * 50)

print("\n[1/7] Sprawdzanie GPU...")
print("GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

BASE_MODEL = "speakleash/Bielik-4.5B-v3"
OUTPUT_DIR = "/kaggle/working/lyra_adapter"

possible_paths = [
    Path("/kaggle/input/datasets/tomaszstraw/lyra-train"),
    Path("/kaggle/input/datasets/tomaszstraw/lyra-train-op"),
    Path("/kaggle/input/lyra-datasets"),
    Path("/kaggle/working/datasets"),
]

DATASETS_DIR = None
for p in possible_paths:
    if p.exists():
        test_files = list(p.glob("*.jsonl"))
        if test_files:
            DATASETS_DIR = p
            print(f"  Znaleziono: {p}")
            break

print("\n[2/7] Szukanie datasetów...")

dataset_files = [
    "00_tozsamosc.jsonl",
    "01_core_modes.jsonl",
    "02_tools.jsonl", 
    "03_memory.jsonl",
    "04_states.jsonl",
    "05_tests.jsonl",
]

datasets = []
loaded_from_kaggle = False

if DATASETS_DIR:
    for fname in dataset_files:
        fpath = DATASETS_DIR / fname
        if fpath.exists():
            ds = load_dataset("json", data_files=str(fpath), split="train")
            datasets.append(ds)
            print(f"  ✓ {fname}: {len(ds)} przykładów")
            loaded_from_kaggle = True

if not loaded_from_kaggle:
    print("  Pobieranie z GitHub...")
    DATASETS_DIR = Path("/kaggle/working/datasets")
    DATASETS_URL = "https://raw.githubusercontent.com/alfik21/lyra-datasets/main"
    
    for fname in dataset_files:
        url = f"{DATASETS_URL}/{fname}"
        fpath = DATASETS_DIR / fname
        print(f"  Pobieranie {fname}...")
        urllib.request.urlretrieve(url, fpath)
        ds = load_dataset("json", data_files=str(fpath), split="train")
        datasets.append(ds)
        print(f"  ✓ {fname}: {len(ds)} przykładów")

dataset = concatenate_datasets(datasets)
print(f"\n  RAZEM: {len(dataset)} przykładów")

print("\n[3/7] Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n[4/7] Model (FP16)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("\n[5/7] Konfiguracja LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def tokenize(example):
    messages = example.get("messages", [])
    text = ""
    
    if not messages:
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        if input_text or output_text:
            text = f"User: {input_text}\nAssistant: {output_text}"
    else:
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                text += f"User: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n"
    
    if not text.strip():
        text = " "
    
    return tokenizer(text, truncation=True, max_length=512)

print("\n[6/7] Tokenizacja datasetu...")
dataset = dataset.map(tokenize, batched=False)

print("\n[7/7] Trening...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    warmup_steps=20,
    optim="adamw_torch",
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("\n" + "=" * 50)
print("START TRENINGU")
print("=" * 50)
trainer.train()

print("\nZapisywanie adaptera...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✓ GOTOWE! Adapter: {OUTPUT_DIR}")
print("\nPobierz: /kaggle/working/lyra_adapter")
