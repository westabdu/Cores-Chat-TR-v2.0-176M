"""
Cores-AI TR - Türkçe Veri Hazırlama Scripti
Veri setlerini tokenize edip train.bin / val.bin oluşturur.

Kullanım:
    python prepare_tr.py
"""

import os
import numpy as np
import pickle
from datasets import load_from_disk
from tokenizers import Tokenizer
import random
import time

# ---- AYARLAR ----
TOKENIZER_PATH = "tokenizer_tr_v2.json"
OUTPUT_DIR = os.path.join("data", "turkce")
TRAIN_RATIO = 0.95

DATASETS = [
    {"path": "veri/wikipedia", "text_col": "text", "name": "Wikipedia"},
    {"path": "veri/matematik", "text_col": "text", "name": "Matematik"},
    {"path": "veri/sohbet",    "text_col": "text", "name": "Sohbet"},
]


def main():
    print("=" * 60)
    print("CORES-AI TR - VERİ HAZIRLAMA")
    print("=" * 60)

    print(f"\n📦 Tokenizer yükleniyor: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Vocab boyutu: {vocab_size}")

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    print(f"   [BOS] ID: {bos_id}, [EOS] ID: {eos_id}")

    assert vocab_size <= 65535, f"Vocab boyutu ({vocab_size}) uint16'ya sığmıyor!"

    all_tokens = []
    total_texts = 0
    total_skipped = 0

    for ds_info in DATASETS:
        ds_path = ds_info["path"]
        text_col = ds_info["text_col"]
        ds_name = ds_info["name"]

        if not os.path.exists(ds_path):
            print(f"\n⚠️  {ds_name} bulunamadı: {ds_path} - atlanıyor")
            continue

        print(f"\n📂 {ds_name} yükleniyor: {ds_path}")
        t0 = time.time()
        dataset = load_from_disk(ds_path)
        print(f"   {len(dataset):,} örnek yüklendi ({time.time()-t0:.1f}s)")

        print(f"   Tokenize ediliyor...")
        t0 = time.time()
        ds_tokens = 0
        ds_texts = 0
        ds_skipped = 0

        for i in range(len(dataset)):
            try:
                text = dataset[i][text_col]
                if not text or not isinstance(text, str) or len(text.strip()) < 20:
                    ds_skipped += 1
                    continue

                encoded = tokenizer.encode(text.strip())
                token_ids = encoded.ids

                if len(token_ids) < 5:
                    ds_skipped += 1
                    continue

                doc_tokens = []
                if bos_id is not None:
                    doc_tokens.append(bos_id)
                doc_tokens.extend(token_ids)
                if eos_id is not None:
                    doc_tokens.append(eos_id)

                all_tokens.extend(doc_tokens)
                ds_tokens += len(doc_tokens)
                ds_texts += 1

            except Exception:
                ds_skipped += 1
                continue

            if (i + 1) % 100000 == 0:
                print(f"   ... {i+1:,} / {len(dataset):,} işlendi ({ds_tokens:,} token)")

        elapsed = time.time() - t0
        print(f"   ✅ {ds_name}: {ds_texts:,} metin → {ds_tokens:,} token ({elapsed:.1f}s, {ds_skipped:,} atlandı)")
        total_texts += ds_texts
        total_skipped += ds_skipped

    print(f"\n{'=' * 60}")
    print(f"TOPLAM: {total_texts:,} metin → {len(all_tokens):,} token")
    print(f"Atlanan: {total_skipped:,}")
    print(f"{'=' * 60}")

    if len(all_tokens) == 0:
        print("❌ Hiç token üretilemedi!")
        return

    print(f"\n✂️  Train/Val bölünüyor ({TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%)...")

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    n = len(all_tokens)
    split_idx = int(n * TRAIN_RATIO)

    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"   Train: {len(train_tokens):,} token ({len(train_tokens)*2/1e6:.1f} MB)")
    print(f"   Val:   {len(val_tokens):,} token ({len(val_tokens)*2/1e6:.1f} MB)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path = os.path.join(OUTPUT_DIR, "val.bin")
    meta_path = os.path.join(OUTPUT_DIR, "meta.pkl")

    print(f"\n💾 Kaydediliyor...")

    train_tokens.tofile(train_path)
    print(f"   ✅ {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB)")

    val_tokens.tofile(val_path)
    print(f"   ✅ {val_path} ({os.path.getsize(val_path)/1e6:.1f} MB)")

    meta = {
        'vocab_size': vocab_size,
        'tokenizer_path': TOKENIZER_PATH,
        'total_tokens': n,
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"   ✅ {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"✅ VERİ HAZIRLAMA TAMAMLANDI!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
