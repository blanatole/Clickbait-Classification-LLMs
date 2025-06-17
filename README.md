# README – Đồ án Phát hiện Clickbait (Xử lý Văn bản)

## 0. Giới thiệu nhanh

* **Bài toán**: phân loại tiêu đề hoặc bài đăng thành **Clickbait** (1) và **Not‑clickbait** (0).
* **Dataset**: Webis‑Clickbait‑17 – đã chia sẵn **80 / 10 / 10**:

  ```
  data/
  ├── train/   # 30 812 mẫu
  ├── val/     # 3 851 mẫu
  └── test/    # 3 854 mẫu
  ```

  Mỗi thư mục chứa `data.jsonl`:

  ```jsonc
  { "id": "abc123", "text": "Tiêu đề", "label": 1 }
  ```

> **Phạm vi**: chỉ xử lý **text** – bỏ `.png`, `.warc`.

---

## 1. Thiết lập môi trường

```bash
conda create -n clickbait python=3.10
conda activate clickbait
pip install -r requirements.txt  # transformers, datasets, peft, unsloth, scikit‑learn, etc.
```

---

## 2. Cấu trúc thư mục dự án

```
project/
├── data/            # mô tả ở trên
├── src/
│   ├── train_bert.py      # fine‑tune BERT / RoBERTa
│   ├── train_deberta.py   # fine‑tune DeBERTa‑v3
│   ├── train_vibert.py    # fine‑tune PhoBERT/viBERT (tiếng Việt)
│   ├── train_lora.py      # fine‑tune LLaMA/Mistral với LoRA
│   ├── eval.py            # đánh giá trên test set
│   └── utils.py
├── outputs/         # checkpoints & logs
├── notebooks/       # (tuỳ chọn) EDA & prompt thử nghiệm
└── README.md
```

---

## 3. Fine‑tuning các mô hình

### 3.1 BERT / RoBERTa (baseline)

* **Yêu cầu**: GPU ≥ 8 GB.
* **Kỳ vọng**: Acc ≈ 85 %, F1 ≈ 0.70.
* **Mã ví dụ**: xem `src/train_bert.py` (đoạn code trong README bản cũ).

### 3.2 DeBERTa‑v3‑base (English)

* **Ưu điểm**: tốt hơn RoBERTa \~1‑2 % F1.
* **Code** (tương tự BERT):

  ```python
  tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
  model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=2)
  ```
* **Tài nguyên**: GPU 10‑12 GB, batch 16, max\_length 128.

### 3.3 DistilBERT / TinyBERT (nhẹ)

* **Dùng khi tài nguyên giới hạn** (≤ 4 GB VRAM).
* **Kỳ vọng**: F1 ≈ 0.63‑0.66.

### 3.4 PhoBERT‑base / viBERT‑base (tiếng Việt)

* **Khi** tiêu đề phần lớn tiếng Việt.
* **Model**: `vinai/phobert-base` hoặc `vibert4news-base-cased`.
* **Chú ý**: chuyển `sentencepiece` + token type xử lý đặc trưng.

### 3.5 ELECTRA‑small (speed‑oriented)

* **Ưu điểm**: infer nhanh gấp \~2× BERT, F1 ≈ 0.67.

### 3.6 LLaMA‑2 / Mistral‑7B với LoRA (PEFT/Unsloth)

* **Yêu cầu**: ≥ 24 GB VRAM (QLoRA: 16 GB).
* **Kỳ vọng**: F1 ≈ 0.72.
* **Script**: `src/train_lora.py`.

> **Mẹo lựa chọn**: nếu GPU nhỏ → DistilBERT; GPU trung bình → DeBERTa; muốn đa nhiệm & giải thích → LLaMA/Mistral LoRA.

---

## 4. Prompting với LLM (không fine‑tune)

### 4.1 Open‑source chat models

| Model                                   | Thông số | Gợi ý sử dụng                                    |
| --------------------------------------- | -------- | ------------------------------------------------ |
| **mistralai/Mistral‑7B‑Instruct**       | 7 B      | Local inference fast, F1 zero‑shot \~0.55        |
| **meta‑llama/Meta‑Llama‑3‑8B‑Instruct** | 8 B      | Few‑shot F1 \~0.65                               |
| **openchat/openchat‑3.5‑1210**          | 7 B      | Thử chain‑of‑thought rồi "Answer: Clickbait/Not" |

### 4.2 Hosted API (chi phí theo token)

| Provider                    | Model                    | Ghi chú                                              |
| --------------------------- | ------------------------ | ---------------------------------------------------- |
| **OpenAI GPT‑4o**           | `gpt-4o-2024-05-13`      | Few‑shot 3‑5 ví dụ, F1 \~0.75                        |
| **Anthropic Claude 3 Opus** | `claude-3-opus-20240229` | Khéo prompt "Respond with one word" để giảm dài dòng |
| **Google Gemini 1.5 Pro**   | `gemini-1.5-pro-latest`  | Cần giới hạn output length                           |

### 4.3 Kỹ thuật prompt

1. **Role + Task + Format** (R‑T‑F).
2. Sử dụng **few‑shot** 2‑5 cặp ví dụ.
3. Yêu cầu **chỉ trả lời** "Clickbait" hoặc "Not clickbait".
4. Với Claude/GPT‑4o: thêm `"You must not provide explanation"` nếu model hay giải thích.

---

## 5. Đánh giá

```bash
python src/eval.py \
  --model_dir outputs/deberta \
  --test_file data/test/data.jsonl
```

`eval.py` in kết quả **Accuracy, Precision, Recall, F1** và confusion matrix.

---

## 6. Báo cáo & Phân tích

* Bảng so sánh mọi mô hình (Fine‑tune vs Prompt).
* Error analysis: show 20 tiêu đề khó, lý do sai.
* Chi phí: GPU giờ, token API.

---

## 7. Ghi chú

* Nếu tiêu đề song ngữ Anh‑Việt, cân nhắc **multilingual XLM‑R‑base**.
* DistilBERT/ELECTRA có thể dùng **knowledge distillation** từ DeBERTa để tăng F1.

---

## 8. Tham khảo

* Webis‑Clickbait‑17 Corpus.
* HuggingFace Transformers, PEFT, Unsloth.
* "PhoBERT: Pre‑trained BERT models for Vietnamese" – Nguyen & al., 2020.
* "DeBERTa: Decoding‑enhanced BERT with disentangled attention" – He & al., 2021.
