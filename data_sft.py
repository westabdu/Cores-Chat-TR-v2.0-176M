"""
Cores-AI TR v2.0 — SFT DataLoader Modülü (data_sft.py)
Aşama 4 & 5 için SFT Veri Yükleyicisi

Bu modül:
1. JSONL formatındaki SFT veri kümelerini (train.jsonl, val.jsonl) okur.
2. Değişken uzunluktaki diyalogları batch içinde dinamik olarak pad eder.
3. Input tokens dizisini pad_id (1) ile doldurur.
4. Target labels dizisini -1 (ignore_index) ile doldurur.
5. Causal LM kaydırmasını uygular: x = tokens[:, :-1], y = labels[:, 1:].
6. block_size=512 sınırına riayet eder.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

# Tokenizer sabitleri
TOKEN_PAD_ID = 1
TOKEN_IGNORE_TARGET = -1
DEFAULT_BLOCK_SIZE = 512


class SFTDataset(Dataset):
    """JSONL formatındaki SFT verilerini belleğe yükleyen Dataset sınıfı."""
    def __init__(self, jsonl_path: str):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"SFT veri dosyası bulunamadı: {jsonl_path}")

        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "tokens" in data and "labels" in data:
                        self.samples.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Hatalı JSON satırı ({line_num}): {e}")

        if len(self.samples) == 0:
            raise ValueError(f"Geçerli SFT örneği bulunamadı: {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class SFTCollateFn:
    """
    Değişken uzunluktaki örnekleri batch içinde hizalayan ve
    causal LM girdi/hedef kaydırmasını (x, y) oluşturan Collate fonksiyonu.
    """
    def __init__(
        self,
        pad_id: int = TOKEN_PAD_ID,
        pad_target: int = TOKEN_IGNORE_TARGET,
        block_size: int = DEFAULT_BLOCK_SIZE
    ):
        self.pad_id = pad_id
        self.pad_target = pad_target
        self.block_size = block_size

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Batch içindeki maksimum uzunluğu bul (block_size ile sınırlandır)
        batch_lengths = [min(len(item["tokens"]), self.block_size) for item in batch]
        max_len = max(batch_lengths)

        padded_tokens = []
        padded_labels = []

        for item in batch:
            toks = item["tokens"][:self.block_size]
            labs = item["labels"][:self.block_size]
            cur_len = len(toks)
            pad_len = max_len - cur_len

            # Tokens dizisini pad_id ile, labels dizisini pad_target (-1) ile doldur
            padded_tokens.append(toks + [self.pad_id] * pad_len)
            padded_labels.append(labs + [self.pad_target] * pad_len)

        tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long)

        # Causal LM Kaydırma (Shift):
        # x: [0 ... T-2] -> Model girdi dizisi
        # y: [1 ... T-1] -> Modelin tahmin etmesi gereken sonraki token hedef dizisi
        x = tokens_tensor[:, :-1].contiguous()
        y = labels_tensor[:, 1:].contiguous()

        return x, y


def create_sft_dataloader(
    jsonl_path: str,
    batch_size: int,
    shuffle: bool = True,
    pad_id: int = TOKEN_PAD_ID,
    pad_target: int = TOKEN_IGNORE_TARGET,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_workers: int = 0
) -> DataLoader:
    """SFT eğitimi ve doğrulaması için hazır DataLoader döndürür."""
    dataset = SFTDataset(jsonl_path)
    collate_fn = SFTCollateFn(pad_id=pad_id, pad_target=pad_target, block_size=block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
