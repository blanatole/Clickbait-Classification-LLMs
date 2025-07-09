import json
import os

INSTRUCTION = "Phân loại tiêu đề này là clickbait hay không. Chỉ trả về: clickbait hoặc non-clickbait."

input_files = {
    "train": "shared/data/train/data.jsonl",
    "val": "shared/data/val/data.jsonl",
    "test": "shared/data/test/data.jsonl"
}

output_files = {
    "train": "shared/data/train_alpaca.jsonl",
    "val": "shared/data/val_alpaca.jsonl",
    "test": "shared/data/test_alpaca.jsonl"
}

label_map = {0: "non-clickbait", 1: "clickbait"}

def convert_to_alpaca(example):
    return {
        "instruction": INSTRUCTION,
        "input": f"Tiêu đề: {example['text']}",
        "output": label_map[int(example['label'])]
    }

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            alpaca_ex = convert_to_alpaca(ex)
            fout.write(json.dumps(alpaca_ex, ensure_ascii=False) + "\n")
    print(f"✅ Đã tạo {output_path}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        if os.path.exists(input_files[split]):
            process_file(input_files[split], output_files[split])
        else:
            print(f"⚠️ Không tìm thấy file {input_files[split]}") 