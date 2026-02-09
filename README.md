# Cybersecurity-4B-AI-Model - Google Colab 版本
# ===========================================

## 🚀 快速開始

### 方法 1: 直接在 Colab 運行 (推薦)

1. 打開 Google Colab: https://colab.research.google.com
2. 新建筆記本
3. 複製以下代碼並運行:

```python
# 克隆項目
!git clone https://github.com/Look0316/Cybersecurity-4B-AI-Model.git
%cd Cybersecurity-4B-AI-Model

# 安裝依賴
!pip install -r requirements-colab.txt -q

# 運行訓練
!python scripts/colab_train.py
```

### 方法 2: 上傳腳本

1. 下載 `scripts/colab_train.py`
2. 上傳到 Google Colab
3. 運行 `python scripts/colab_train.py`

---

## 📋 Colab vs 本地配置

| 配置項 | Colab (免費) | Colab Pro | 本地 (3060ti) |
|--------|--------------|-----------|---------------|
| GPU | T4/P100 (15GB) | A100 (40GB) | 3060ti (8GB) |
| VRAM | ~14GB | ~40GB | ~7GB |
| 訓練時間 | 2-3 小時 | 1-2 小時 | 3-4 小時 |
| 免費額度 | 每天 12 小時 | 每天 24 小時 | 無限 |
| 數據持久化 | Google Drive | Google Drive | 本地磁盤 |

---

## 🎯 Colab 訓練步驟

### 1. 選擇 GPU
- Runtime → Change runtime type → GPU (T4)

### 2. 掛載 Google Drive
腳本會自動提示掛載，選擇 "連接"

### 3. 運行腳本
```bash
python scripts/colab_train.py
```

### 4. 下載模型
訓練完成後，模型會保存在:
- `/content/drive/MyDrive/Cybersecurity-4B-AI-Model/outputs/cyber-4b-qlora/`

---

## 📦 依賴 (requirements-colab.txt)

```
transformers>=4.40.0
torch>=2.1.0
accelerate>=0.28.0
peft>=0.10.0
bitsandbytes>=0.41.0
trl>=0.8.0
scikit-learn
tqdm
datasets
faiss-cpu
sentence-transformers
google-colab
```

---

## 🔧 常見問題

### Q: GPU 內存不足？
A: Colab T4 有 15GB VRAM，足够運行 QLoRA

### Q: 訓練中斷怎麼辦？
A: 使用 Google Drive 保存 checkpoint，從上次位置繼續

### Q: 如何查看訓練進度？
A: 腳本會實時打印 loss 和進度

### Q: 訓練完成後如何測試？
A:
```python
from scripts.test_tinyllm import test_model
test_model("outputs/cyber-4b-qlora")
```

---

## 📊 預期輸出

```
============================================================
🔐 CyberSec 4B Model - Colab Training
============================================================

📂 掛載 Google Drive...
✅ 項目路徑: /content/drive/MyDrive/Cybersecurity-4B-AI-Model

📦 安裝依賴...
✅ 依賴安裝完成

🔍 GPU 診斷
============================================================
✅ GPU: Tesla T4
   總記憶體: 14.75 GB
   已分配: 0.50 GB
   可用: 14.25 GB

📊 推薦配置:
   Batch Size: 4
   Gradient Accumulation: 4
   Effective Batch: 16

📝 生成 TinyLLM 數據...
   模型: Qwen/Qwen2.5-7B-Instruct
   樣本數: 2000
   100%|████████| 63/63 [05:23<00:00]

✅ 數據已保存: /content/drive/MyDrive/.../data/distilled_tinyllm.jsonl

🚀 開始 QLoRA 訓練...
   數據: data/distilled_tinyllm.jsonl
   輸出: outputs/cyber-4b-qlora
   Epochs: 3

🔥 開始訓練...
Training: 100%|████████| 500/500 [01:32<00:00]

✅ 模型已保存: outputs/cyber-4b-qlora

🎉 訓練完成!
📁 模型位置: outputs/cyber-4b-qlora
📁 數據位置: data/distilled_tinyllm.jsonl
```

---

## 🏠 本地版本 vs Colab 版本

| 功能 | 本地版本 | Colab 版本 |
|------|---------|-----------|
| 數據生成 | ✅ | ✅ |
| QLoRA 訓練 | ✅ | ✅ |
| RAG 模塊 | ✅ (CPU) | ✅ (GPU) |
| Google Drive 持久化 | ❌ | ✅ |
| 免費 GPU | ❌ | ✅ |
| 4-bit 嵌入模型 | 推薦 | 自動 |

---

## 📝 下一步

1. **在 Colab 運行**: 獲取訓練好的模型
2. **本地推理**: 使用 transformers 載入模型
3. **RAG 增強**: 添加最新 CVE 數據
4. **部署**: Docker + API 服務
