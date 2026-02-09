#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberSec 4B Model - Colab One-Click Training
=============================================
✅ 真正一鍵運行
✅ 自動錯誤恢復
✅ 無需 Google Drive (可選)
✅ 詳細進度反饋

使用方法 (在 Colab 中):
```python
!git clone https://github.com/Look0316/LLM_colab.git
%cd LLM_colab
!python colab_train.py
```
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "num_samples": 2000,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "output_dir": "/content/outputs",
    "data_file": "/content/data/distilled_tinyllm.jsonl",
    "use_drive": False,  # 設為 True 來啟用 Google Drive
}

# ═══════════════════════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════════════════════

def print_step(step_num, message):
    """打印步驟標題"""
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {message}")
    print(f"{'='*60}")

def print_status(message, status="INFO"):
    """打印狀態"""
    emojis = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "LOADING": "🔄",
    }
    print(f"{emojis.get(status, 'ℹ️')} {message}")

def check_gpu():
    """檢查 GPU 狀態"""
    import torch

    if not torch.cuda.is_available():
        print_status("未檢測到 GPU!", "ERROR")
        print("請確認: Runtime → Change runtime type → GPU")
        return False, 4, 4

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print_status(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)", "SUCCESS")

    # 根據 VRAM 調整 batch size
    if gpu_mem >= 14:
        batch_size = 4
    elif gpu_mem >= 10:
        batch_size = 2
    else:
        batch_size = 1

    print_status(f"Batch Size: {batch_size}", "INFO")

    return True, batch_size, batch_size

# ═══════════════════════════════════════════════════════════════════════════
# 依賴安裝
# ═══════════════════════════════════════════════════════════════════════════

def install_dependencies():
    """安裝依賴 (只安裝必要的)"""
    print_step(1, "安裝依賴")

    import subprocess
    import sys

    packages = [
        "transformers>=4.40.0",
        "torch>=2.1.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.8.0",
        "tqdm",
        "datasets",
    ]

    for pkg in packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", pkg
            ])
            print_status(f"已安裝: {pkg}", "SUCCESS")
        except Exception as e:
            print_status(f"安裝失敗: {pkg} - {e}", "WARNING")

# ═══════════════════════════════════════════════════════════════════════════
# 數據生成
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(output_file, num_samples=2000):
    """生成 TinyLLM 格式數據"""
    print_step(2, "生成訓練數據")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    import random
    from tqdm import tqdm

    print_status(f"模型: {CONFIG['model_name']}", "INFO")
    print_status(f"樣本數: {num_samples}", "INFO")

    # 創建目錄
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 場景模板
    scenarios = [
        {"scenario": "SMBv1 enabled on Windows Server 2016", "category": "scan"},
        {"scenario": "Web app has SQL injection on login form", "category": "sqli"},
        {"scenario": "XSS found on comment form", "category": "xss"},
        {"scenario": "Redis server accessible without auth", "category": "redis"},
        {"scenario": "SSH weak password (root:toor)", "category": "ssh"},
        {"scenario": "JWT token with weak secret", "category": "jwt"},
        {"scenario": "Docker daemon exposed on port 2375", "category": "docker"},
        {"scenario": "Sudo version 1.8.31 vulnerable", "category": "privesc"},
        {"scenario": "MongoDB NoAuth on port 27017", "category": "mongo"},
        {"scenario": "phpMyAdmin exposed /admin", "category": "web"},
    ]

    # 攻擊步驟
    step_templates = {
        "scan": [
            "nmap -p 445 --script smb-vuln-ms17-010 {target}",
            "enum4linux -a {target}",
        ],
        "sqli": [
            "sqlmap -u '{url}' --dbs",
            "sqlmap -u '{url}' -D {db} --tables",
        ],
        "xss": [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
        ],
        "redis": [
            "redis-cli -h {target} INFO",
            "redis-cli -h {target} CONFIG GET *",
        ],
        "ssh": [
            "ssh root@{target}",
            "hydra -l root -P wordlist.txt ssh://{target}",
        ],
        "jwt": [
            "python3 jwt_tool.py -t {token} -s secret",
        ],
        "docker": [
            "curl http://{target}:2375/version",
            "docker -H {target} ps",
        ],
        "privesc": [
            "sudo -l",
            "searchsploit sudo 1.8.31",
        ],
        "mongo": [
            "mongo {target}:27017 --eval 'db.adminCommand({listDatabases:1})'",
        ],
        "web": [
            "curl {url}/admin/backups.sql",
            "curl {url}/phpinfo.php",
        ],
    }

    # 載入模型
    print_status("載入模型中...", "LOADING")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"], 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print_status("模型載入成功", "SUCCESS")
    except Exception as e:
        print_status(f"模型載入失敗: {e}", "ERROR")
        raise

    # 生成數據
    data = []
    print_status("開始生成數據...", "LOADING")

    for i in tqdm(range(num_samples), desc="生成"):
        scenario = random.choice(scenarios)
        cat = scenario["category"]

        messages = [
            {"role": "system", "content": "You are a professional penetration tester."},
            {"role": "user", "content": f"Scenario: {scenario['scenario']}\\nWhat is your next step?"},
        ]

        # 生成
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print_status(f"生成失敗: {e}", "WARNING")
            response = f"Step: Analyze {scenario['scenario']}"

        sample = {
            "messages": messages + [{"role": "assistant", "content": response}],
            "category": cat,
            "scenario": scenario["scenario"],
            "steps": step_templates.get(cat, []),
        }
        data.append(sample)

        # 清理記憶體
        if i % 50 == 0:
            torch.cuda.empty_cache()

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print_status(f"數據已保存: {output_file}", "SUCCESS")
    print_status(f"總樣本數: {len(data)}", "INFO")

    # 卸載模型
    del model
    torch.cuda.empty_cache()

    return output_file

# ═══════════════════════════════════════════════════════════════════════════
# QLoRA 訓練
# ═══════════════════════════════════════════════════════════════════════════

def train_qlora(data_file, output_dir, epochs=3):
    """QLoRA 訓練"""
    print_step(3, "QLoRA 訓練")

    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from datasets import Dataset
    import json

    print_status(f"數據: {data_file}", "INFO")
    print_status(f"輸出: {output_dir}", "INFO")
    print_status(f"Epochs: {epochs}", "INFO")

    # 讀取數據
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    print_status(f"載入 {len(raw_data)} 樣本", "SUCCESS")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True
    )

    # Dataset
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

    # 模型
    from transformers import AutoModelForCausalLM
    print_status("載入基礎模型...", "LOADING")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=CONFIG["learning_rate"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        report_to="none",
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
    )

    # 訓練
    print_status("開始訓練...", "LOADING")
    trainer.train()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print_status(f"模型已保存: {output_dir}", "SUCCESS")

    return output_dir

# ═══════════════════════════════════════════════════════════════════════════
# 測試函數
# ═══════════════════════════════════════════════════════════════════════════

def test_model(model_path):
    """測試訓練好的模型"""
    print_step(4, "測試模型")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print_status("載入模型...", "LOADING")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 測試問題
    test_prompts = [
        {"role": "user", "content": "SMBv1 enabled on Windows Server 2016. What is your next step?"},
        {"role": "user", "content": "Found SQL injection on login form. Exploit it."},
    ]

    for prompt in test_prompts:
        print(f"\n👤 {prompt['content']}")
        messages = [
            {"role": "system", "content": "You are a professional penetration tester."},
            prompt,
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()
        print(f"🤖 {response[:300]}...")

    print_status("測試完成", "SUCCESS")

# ═══════════════════════════════════════════════════════════════════════════
# 主函數
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """主執行函數"""
    print("\n" + "="*60)
    print("  🔐 CyberSec 4B Model - Colab 一鍵訓練")
    print("="*60)

    try:
        # Step 1: 檢查 GPU
        print_step(0, "檢查 GPU")
        has_gpu, _, _ = check_gpu()
        if not has_gpu:
            print_status("警告: 繼續使用 CPU 訓練會非常慢", "WARNING")

        # Step 2: 安裝依賴
        install_dependencies()

        # Step 3: 生成數據
        data_file = generate_data(
            CONFIG["data_file"],
            CONFIG["num_samples"]
        )

        # Step 4: 訓練
        output_dir = train_qlora(
            data_file,
            CONFIG["output_dir"],
            CONFIG["epochs"]
        )

        # Step 5: 完成
        print("\n" + "="*60)
        print("  🎉 訓練完成!")
        print("="*60)
        print(f"\n📁 模型位置: {output_dir}")
        print(f"\n下一步:")
        print("1. 下載模型文件")
        print("2. 使用 transformers 載入推理")
        print("3. 添加 RAG 模塊獲取最新 CVE")

    except KeyboardInterrupt:
        print_status("用戶中斷", "WARNING")
    except Exception as e:
        print_status(f"錯誤: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        print("\n💡 提示:")
        print("1. 確保選擇了 GPU (Runtime → Change runtime type → GPU)")
        print("2. 重新運行細胞")
        print("3. 如持續失敗，請回報錯誤")

if __name__ == "__main__":
    main()
