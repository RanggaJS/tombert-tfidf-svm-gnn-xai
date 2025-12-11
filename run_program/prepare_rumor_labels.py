# coding=utf-8
"""
Utility untuk membuat label rumor vs non-rumor secara heuristik dari dataset Twitter2015.

Cara pakai (tidak menjalankan model lain):
    python3 run_program/prepare_rumor_labels.py

Output:
    output/rumor_labeled/twitter2015_train_rumor.tsv
    output/rumor_labeled/twitter2015_dev_rumor.tsv
    output/rumor_labeled/twitter2015_test_rumor.tsv
    output/rumor_labeled/twitter2015_rumor_all.tsv   (gabungan)

Skema heuristik (bisa diubah di KEYWORDS / STRONG_HINTS):
    - rumor=1 jika mengandung kata/ekspresi yang rawan rumor/berita tak pasti:
      "rumor", "hoax", "fake", "unconfirmed", "alleged", "breaking",
      "explosion", "attack", "shoot", "shot", "killed", "bomb", "??", "!!!"
    - Selain itu diberi label 0 (non-rumor).

Catatan:
    - Heuristik ini sederhana; silakan tambahkan/kurangi kata kunci sesuai kebutuhan.
    - Gunakan hasil ini untuk melatih/pengujian GNN rumor detection.
"""
import csv
import re
from pathlib import Path


DATA_DIR = Path("absa_data/twitter2015")
OUTPUT_DIR = Path("output/rumor_labeled")

# Kata kunci rumor (case-insensitive regex)
KEYWORDS = [
    r"rumor", r"hoax", r"fake", r"unconfirmed", r"alleged", r"breaking",
    r"explosion", r"attack", r"shoot", r"shot", r"killed", r"bomb",
]

# Pola tanda tanya/seru beruntun yang sering muncul di rumor/spekulasi
STRONG_HINTS = [
    r"\?{2,}",   # ??, ???, dst
    r"!{2,}",    # !!, !!!
]


def load_tsv(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            # index, label, image_id, text, target
            rows.append({
                "text": parts[3],
                "raw": parts,
            })
    return rows


def label_rumor(text: str) -> int:
    t = text.lower()
    for pat in KEYWORDS + STRONG_HINTS:
        if re.search(pat, t):
            return 1
    return 0


def write_tsv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["text", "label_rumor"])
        for r in rows:
            w.writerow([r["text"], r["label_rumor"]])


def main():
    splits = {
        "train": DATA_DIR / "train.tsv",
        "dev": DATA_DIR / "dev.tsv",
        "test": DATA_DIR / "test.tsv",
    }

    all_rows = []

    for split, path in splits.items():
        if not path.exists():
            print(f"[WARN] File tidak ditemukan: {path}")
            continue
        rows = load_tsv(path)
        for r in rows:
            r["label_rumor"] = label_rumor(r["text"])
        out_path = OUTPUT_DIR / f"twitter2015_{split}_rumor.tsv"
        write_tsv(out_path, rows)
        all_rows.extend(rows)
        print(f"[OK] {split}: tulis {len(rows)} baris -> {out_path}")

    if all_rows:
        out_all = OUTPUT_DIR / "twitter2015_rumor_all.tsv"
        write_tsv(out_all, all_rows)
        print(f"[OK] gabungan: {len(all_rows)} baris -> {out_all}")
    else:
        print("[WARN] Tidak ada data yang diproses.")


if __name__ == "__main__":
    main()

