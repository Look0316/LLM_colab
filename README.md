# Cybersecurity-4B-AI-Model - Google Colab 版本
# ===========================================

## 🚀 快速開始

### 方法 1: 直接在 Colab 運行 (推薦)

1. 打開 Google Colab: https://colab.research.google.com
2. 新建筆記本 (Python 3)
3. 複製以下代碼並運行:

```python
# 克隆項目
!git clone https://github.com/Look0316/LLM_colab.git
%cd LLM_colab

# 安裝依賴
!pip install -r requirements-colab.txt -q

# 運行完整訓練流程
!python colab_complete.py
```

### 方法 2: 手動下載上傳

1. 下載 `colab_complete.py`
2. 上傳到 Google Colab
3. 運行 `!python colab_complete.py`

---

## 🎯 Colab 訓練步驟

### 1. 選擇 GPU
- Runtime → Change runtime type → **GPU (T4)**

### 2. 運行腳本

```python
!git clone https://github.com/Look0316/LLM_colab.git
%cd LLM_colab
!pip install -r requirements-colab.txt -q
!python colab_complete.py
```

### 3. 等待完成
- Step 1: 生成 2000 條 TinyLLM 數據 (~5 分鐘)
- Step 2: QLoRA 訓練 (~1.5 小時)
- Step 3: 測試模型 (~1 分鐘)

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
```

---

## 📊 預期輸出

```
============================================================
  🔐 Cybersecurity 4B Model - Colab 完整訓練流程
============================================================

Step 0: 檢查 GPU
  ✅ GPU: Tesla T4 (14.7 GB)

Step 1: 安裝依賴
  ✅ 依賴安裝完成

Step 2: 生成 TinyLLM 格式數據
  100%|████████| 2000/2000 [05:00]
  ✅ 數據已保存

Step 3: QLoRA 訓練
  100%|████████| 500/500 [01:30]
  ✅ 模型已保存

Step 4: 測試模型
  ✅ 測試完成

🎉 訓練流程完成!
📁 模型位置: /content/outputs/finetuned_tinyllm_v1
```

---

## 📁 下載模型

訓練完成後，模型保存在:
```
/content/outputs/finetuned_tinyllm_v1/
├── adapter_config.json
├── adapter_model.bin
├── tokenizer.json
└── ...
```

**下載方法:**
- 右鍵點擊文件夾 → Download
- 或使用代碼:
```python
from google.colab import files
files.download('/content/outputs/finetuned_tinyllm_v1/adapter_model.bin')
```
