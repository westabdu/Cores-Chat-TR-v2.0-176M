"""
Cores-AI TR v2.0 — SFT Veri Hazırlama Betiği (prepare_sft.py)
Aşama 5B: Gerçek Açık Lisanslı Veri Kümelerinden Pilot SFT Verisi Hazırlama

Bu betik:
1. 3 onaylı açık lisanslı veri kümesinden (sixfingerdev, ogulcanaydogan, AlicanKiraz0)
   satır satır güvenli şekilde veri okur.
2. Alanları Kullanıcı/Asistan formatına eşler (prompt/completion, user/assistant, input/output).
3. Kalite filtreleme (boş/kısa metinler) ve tam tekilleştirme (deduplication) uygular.
4. benchmark_questions.json ve eval_v1_baseline.json ile çakışan soruları (veri sızıntısı) eler.
   Benchmark dosyası bulunamazsa güvenlik gereği işlemi anında durdurur.
5. [BOS]Kullanıcı: ...\nAsistan: ...[EOS] şablonuyla tokenize eder.
6. Loss maskesi oluşturur: Kullanıcı istemi -1 ile maskelenir, asistan yanıtı
   ve [EOS] tokenı eğitime dahil edilir (ignore_index = -1).
7. 512 token sınırını aşan örnekleri sessizce kesmek yerine filtreler ve istatistiklere ekler.
8. Train (%90) ve Validation (%10) setlerine böler, aralarında sıfır çakışma olduğunu doğrular.
9. Sonuçları data/sft_pilot/ altında train.jsonl, val.jsonl ve detaylı dataset_stats.json olarak kaydeder.
"""

import os
import sys
import json
import re
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any
from tokenizers import Tokenizer

# Windows konsolunda UTF-8 çıktı güvencesi
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Sabit Özel Token Kimlikleri (tokenizer_tr_v2.json ile uyumlu)
TOKEN_UNK_ID = 0
TOKEN_PAD_ID = 1
TOKEN_BOS_ID = 2
TOKEN_EOS_ID = 3

# 10 Adet Sentetik Test Verisi (Aşama 3A / Hızlı Doğrulama için)
SYNTHETIC_DATA_SAMPLES = [
    {
        "category": "Selamlaşma",
        "instruction": "Merhaba!",
        "response": "Merhaba! Size nasıl yardımcı olabilirim?"
    },
    {
        "category": "Rol Tanıma",
        "instruction": "Sen kimsin?",
        "response": "Ben Cores-AI, Türkçe için geliştirilmiş bir yapay zeka asistanıyım."
    },
    {
        "category": "Hal-Hatır",
        "instruction": "Nasılsın?",
        "response": "İyiyim, teşekkür ederim. Sizin için ne yapabilirim?"
    },
    {
        "category": "Faktüel Bilgi",
        "instruction": "Dünyanın en yüksek dağı hangisidir?",
        "response": "Dünyanın en yüksek dağı, deniz seviyesinden 8.848 metre yükseklikte olan Everest Dağı'dır."
    },
    {
        "category": "Faktüel Bilgi",
        "instruction": "Işık hızı yaklaşık ne kadardır?",
        "response": "Işık hızı boşlukta yaklaşık olarak saniyede 300.000 kilometredir (tam olarak 299.792.458 m/s)."
    },
    {
        "category": "Talimat Takibi",
        "instruction": "Sağlıklı bir yaşam için 3 önemli alışkanlık listele.",
        "response": "1. Düzenli ve dengeli beslenmek.\n2. Günde en az 7-8 saat kaliteli uyumak.\n3. Günlük düzenli fiziksel aktivite veya egzersiz yapmak."
    },
    {
        "category": "Talimat Takibi",
        "instruction": "Sadece Evet ya da Hayır olarak cevap ver: Penguenler uçabilir mi?",
        "response": "Hayır."
    },
    {
        "category": "Kısa Özetleme",
        "instruction": "Şu cümleyi tek bir ana fikirle açıkla: Fotosentez, bitkilerin güneş ışığını kullanarak besin ve oksijen üretmesidir.",
        "response": "Bitkiler güneş ışığı sayesinde kendi besinlerini üretir ve doğaya oksijen salar."
    },
    {
        "category": "Türkçe Açıklama",
        "instruction": "Deyim nedir, kısa bir örnek ver.",
        "response": "Deyim, genellikle gerçek anlamından farklı bir anlam taşıyan kalıplaşmış söz öbeğidir. Örneğin: 'Göze girmek' ilgi ve sevgi kazanmak demektir."
    },
    {
        "category": "Kapatma / Teşekkür",
        "instruction": "Yardımın için çok teşekkür ederim.",
        "response": "Rica ederim! Başka bir sorunuz olursa her zaman yardımcı olmaktan memnuniyet duyarım."
    }
]


def normalize_text_for_match(text: str) -> str:
    """Metin karşılaştırması ve sızıntı kontrolü için normalizasyon."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def load_benchmark_leak_filters(benchmark_path: str, eval_baseline_path: str = "eval_v1_baseline.json") -> set:
    """
    benchmark_questions.json ve eval_v1_baseline.json dosyalarından test sorularını okur ve
    eğitim verisine sızmasını engellemek için bir küme (set) oluşturur.
    Kritik kural: Benchmark dosyası bulunamazsa veya boşsa hata verir ve durdurur!
    """
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(
            f"❌ Kritik Hata: Benchmark sızıntı koruma dosyası bulunamadı: '{benchmark_path}'! "
            f"Veri sızıntısı izolasyonu garanti edilemediği için işlem durduruldu."
        )

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"❌ Kritik Hata: Benchmark dosyasında ('{benchmark_path}') hiçbir soru bulunamadı!")

    benchmark_prompts = set()
    for q in questions:
        prompt = q.get("prompt", "")
        if prompt:
            benchmark_prompts.add(normalize_text_for_match(prompt))

    # Varsa eval_v1_baseline.json içerisindeki promptları da ekle
    if os.path.exists(eval_baseline_path):
        try:
            with open(eval_baseline_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            for item in eval_data.get("results", []):
                p = item.get("prompt", "")
                if p:
                    benchmark_prompts.add(normalize_text_for_match(p))
        except Exception as e:
            print(f"⚠️ Bilgi: eval_v1_baseline.json okunurken hata: {e}")

    print(f"🔒 Benchmark sızıntı filtresi yüklendi: {len(benchmark_prompts)} soru korunuyor.")
    return benchmark_prompts


def extract_qa_pair(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Farklı Türkçe veri seti formatlarından (sixfingerdev, AlicanKiraz0, ogulcanaydogan)
    kullanıcı talimatı, asistan yanıtı ve sistem promptunu çıkarır.
    Dönüş: (user_text, asst_text, system_text)
    """
    system_text = record.get("system", "").strip() if isinstance(record.get("system"), str) else ""

    # Format 1: ogulcanaydogan ('prompt', 'completion')
    if "prompt" in record and "completion" in record:
        user_text = str(record["prompt"]).strip()
        asst_text = str(record["completion"]).strip()
        return user_text, asst_text, system_text

    # Format 2: sixfingerdev / alpaca ('input', 'output')
    if "input" in record and "output" in record:
        inp = str(record.get("input", "")).strip()
        ins = str(record.get("instruction", "")).strip()
        out = str(record.get("output", "")).strip()
        if ins and inp and ins != inp:
            user_text = f"{ins}\n{inp}"
        else:
            user_text = ins if ins else inp
        return user_text, out, system_text

    # Format 3: AlicanKiraz0 / ShareGPT ('user', 'assistant')
    if "user" in record and "assistant" in record:
        user_text = str(record["user"]).strip()
        asst_text = str(record["assistant"]).strip()
        return user_text, asst_text, system_text

    # Format 4: ogulcanaydogan alternatif ('instruction', 'completion')
    if "instruction" in record and "completion" in record:
        user_text = str(record["instruction"]).strip()
        asst_text = str(record["completion"]).strip()
        return user_text, asst_text, system_text

    # Format 5: Standart ('instruction', 'response')
    if "instruction" in record and "response" in record:
        user_text = str(record["instruction"]).strip()
        asst_text = str(record["response"]).strip()
        return user_text, asst_text, system_text

    return None, None, None


def format_and_tokenize(
    user_text: str,
    asst_text: str,
    tokenizer: Tokenizer,
    max_length: int = 512
) -> Tuple[Optional[Dict[str, Any]], str, int]:
    """
    Kullanıcı ve asistan metnini tokenize eder ve loss maskesini oluşturur.

    Şablon:
    [BOS]Kullanıcı: {user_text}\nAsistan: {asst_text}[EOS]

    Kural:
    - prompt_prefix = '[BOS]Kullanıcı: {user_text}\nAsistan:'
    - full_text = '[BOS]Kullanıcı: {user_text}\nAsistan: {asst_text}[EOS]'
    - prefix_len tokenlarına karşılık gelen labels elemanları: -1 (maskeli)
    - asistan yanıtı ve [EOS] tokenlarına karşılık gelen labels: gerçek token ID
    - Toplam uzunluk > max_length ise sessiz kesim yapılmaz, filtrelenir.
    """
    user_clean = user_text.strip()
    asst_clean = asst_text.strip()

    if not user_clean or not asst_clean:
        return None, "EMPTY_TEXT", 0

    prompt_prefix = f"[BOS]Kullanıcı: {user_clean}\nAsistan:"
    full_text = f"[BOS]Kullanıcı: {user_clean}\nAsistan: {asst_clean}[EOS]"

    prompt_ids = tokenizer.encode(prompt_prefix).ids
    full_ids = tokenizer.encode(full_text).ids
    prefix_len = len(prompt_ids)
    total_len = len(full_ids)

    # 1. Uzunluk kontrolü: 512 sınırını aşarsa filtrele
    if total_len > max_length:
        return None, "TOO_LONG", total_len

    # 2. [EOS] kontrolü: Son token kesinlikle EOS_ID (3) olmalı
    if total_len == 0 or full_ids[-1] != TOKEN_EOS_ID:
        return None, "NO_EOS_AT_END", total_len

    # 3. Prefix eşleşme güvencesi
    if full_ids[:prefix_len] != prompt_ids:
        # BPE sınır anomalisi kontrolü: İki nokta tokenını ara
        colon_id = tokenizer.token_to_id(":")
        found_prefix = False
        for split_pos in range(min(prefix_len - 2, 5), min(prefix_len + 3, total_len)):
            if full_ids[split_pos - 1] == colon_id:
                prefix_len = split_pos
                found_prefix = True
                break
        if not found_prefix:
            return None, "PREFIX_MISMATCH", total_len

    # 4. Labels dizisi oluşturma (Loss Masking):
    # Kullanıcı bölümü ve ayraçlar -> -1 (CrossEntropyLoss ignore_index=-1 ile eğitime katılmaz)
    # Asistan yanıtı ve [EOS] -> gerçek token ID (eğitilir)
    labels = [-1] * prefix_len + list(full_ids[prefix_len:])

    assert len(labels) == len(full_ids), "Tokens ve labels uzunlukları eşit olmalıdır!"

    sample = {
        "tokens": full_ids,
        "labels": labels,
        "length": total_len,
        "prefix_len": prefix_len,
        "asst_len": total_len - prefix_len,
        "user_text": user_clean,
        "asst_text": asst_clean
    }
    return sample, "OK", total_len


def print_alignment_verification(sample: Dict[str, Any], tokenizer: Tokenizer):
    """
    Loss maskeleme ve next-token kaydırma mantığını görsel ve teknik olarak doğrular.
    """
    tokens = sample["tokens"]
    labels = sample["labels"]
    prefix_len = sample["prefix_len"]

    x = tokens[:-1]
    y = labels[1:]

    print("\n" + "=" * 80)
    print("🔍 TEKNİK DOĞRULAMA: TOKEN VE LOSS HİZALAMA TABLOSU")
    print("=" * 80)
    print(f"Toplam Token Sayısı : {len(tokens)}")
    print(f"Kullanıcı İstemi    : {prefix_len} token (Maskelenen, loss=-1)")
    print(f"Asistan Yanıtı + EOS: {len(tokens) - prefix_len} token (Eğitilen, loss=aktif)")
    print("-" * 80)
    print(f"{'İndeks':<7} | {'Model Girdisi (x)':<20} | {'Hedef Token (y)':<20} | {'Loss Durumu'}")
    print("-" * 80)

    for i in range(len(x)):
        x_tok_id = x[i]
        y_tok_id = y[i]

        x_str = repr(tokenizer.decode([x_tok_id])) if x_tok_id != TOKEN_BOS_ID else "'[BOS]'"
        if y_tok_id == -1:
            y_str = "<MASKE: -1>"
            status = "🔴 ATLA (Loss yok)"
        elif y_tok_id == TOKEN_EOS_ID:
            y_str = "'[EOS]' (3)"
            status = "🟢 EĞİTİM (Bitiş Durma)"
        else:
            y_str = repr(tokenizer.decode([y_tok_id]))
            status = "🟢 EĞİTİM (Asistan Yanıtı)"

        if i == prefix_len - 1:
            status += " ⭐️ [İLK ASİSTAN TOKENI]"

        print(f"{i:<7} | {x_tok_id:<5} {x_str:<14} | {y_tok_id:<5} {y_str:<14} | {status}")

    print("=" * 80)
    print("✅ Gözlem Özeti:")
    print("   1. [BOS] ve 'Kullanıcı: ... \nAsistan:' girdilerinde Hedef = -1 (Loss sıfır).")
    print(f"   2. Model tam olarak indeks {prefix_len - 1}'deki ':' tokenını gördüğünde asistanın İLK tokenını tahmin etmeye başlar.")
    print("   3. Asistan yanıtının son tokenında Hedef = [EOS] (3) olur; model düzgün durmayı öğrenir.")
    print("=" * 80 + "\n")


def build_synthetic_dataset(
    tokenizer: Tokenizer,
    benchmark_filters: set,
    max_length: int = 512
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aşama 3A test planı için 10 sentetik örnekten oluşan veri setini işler."""
    processed = []
    skipped_leak = 0
    skipped_length = 0

    print(f"🔨 Sentetik test veri kümesi işleniyor ({len(SYNTHETIC_DATA_SAMPLES)} örnek)...")

    for item in SYNTHETIC_DATA_SAMPLES:
        user_text, asst_text = item["instruction"], item["response"]

        norm_prompt = normalize_text_for_match(user_text)
        if norm_prompt in benchmark_filters:
            print(f"   ⚠️ Benchmark sızıntısı atlandı: '{user_text}'")
            skipped_leak += 1
            continue

        sample, status, length = format_and_tokenize(user_text, asst_text, tokenizer, max_length=max_length)
        if status == "OK":
            processed.append(sample)
        elif status == "TOO_LONG":
            skipped_length += 1
            print(f"   ⚠️ Uzunluk aşıldı ({length} > {max_length}): '{user_text}'")
        else:
            print(f"   ⚠️ Hata ({status}): '{user_text}'")

    audit_stats = {
        "synthetic": {
            "read": len(SYNTHETIC_DATA_SAMPLES),
            "accepted": len(processed),
            "skipped_leak": skipped_leak,
            "skipped_length": skipped_length,
            "skipped_dup": 0
        }
    }
    return processed, audit_stats


def build_full_dataset(
    tokenizer: Tokenizer,
    benchmark_filters: set,
    max_length: int = 512,
    target_qa_count: int = 5000,
    target_instruct_count: int = 3500,
    target_dialog_count: int = 2000
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Onaylanan açık lisanslı 3 Türkçe veri kaynağından dengeli SFT karışımını satır satır işleyerek oluşturur:
    1. sixfingerdev/turkish-qa-multi-dialog-dataset (MIT) -> Hedef: ~5.000 faktüel QA
    2. ogulcanaydogan/Turkish-LLM-v10-Training (Apache 2.0) -> Hedef: ~3.500 talimat
    3. AlicanKiraz0/Turkish-SFT-Dataset-v1.0 (MIT) -> Hedef: ~2.000 temiz diyalog

    Filtreleme Aşamaları:
    - Boş, aşırı kısa veya bozuk metinler (< 4 karakter)
    - Yinelenen (duplicate) kayıtlar (hash tablosuyla)
    - benchmark_questions.json ile çakışan (sızıntı) sorular
    - 512 token sınırını aşan metinler (kesilmez, elenir)
    """
    from huggingface_hub import hf_hub_download

    processed = []
    seen_hashes = set()

    audit_stats = {
        "sixfingerdev": {
            "license": "MIT",
            "read": 0, "accepted": 0,
            "skipped_empty": 0, "skipped_dup": 0,
            "skipped_leak": 0, "skipped_length": 0
        },
        "ogulcanaydogan": {
            "license": "Apache-2.0",
            "read": 0, "accepted": 0,
            "skipped_empty": 0, "skipped_dup": 0,
            "skipped_leak": 0, "skipped_length": 0
        },
        "AlicanKiraz0": {
            "license": "MIT",
            "read": 0, "accepted": 0,
            "skipped_empty": 0, "skipped_dup": 0,
            "skipped_leak": 0, "skipped_length": 0,
            "has_system_count": 0,
            "generic_system_omitted": 0
        }
    }

    print("\n📦 HuggingFace Açık Veri Kaynakları Satır Satır İşleniyor...")

    # 1. Kaynak: sixfingerdev/turkish-qa-multi-dialog-dataset (MIT)
    try:
        print(f"\n[1/3] İşleniyor: sixfingerdev/turkish-qa-multi-dialog-dataset (Hedef: ~{target_qa_count})...")
        p1 = hf_hub_download("sixfingerdev/turkish-qa-multi-dialog-dataset", "data.json", repo_type="dataset")
        with open(p1, "r", encoding="utf-8") as f:
            raw_qa = json.load(f)

        for item in raw_qa:
            if audit_stats["sixfingerdev"]["accepted"] >= target_qa_count:
                break
            audit_stats["sixfingerdev"]["read"] += 1

            u, a, _ = extract_qa_pair(item)
            if not u or not a or len(u) < 4 or len(a) < 4:
                audit_stats["sixfingerdev"]["skipped_empty"] += 1
                continue

            h = hash(u.strip().lower() + "|||" + a.strip().lower())
            if h in seen_hashes:
                audit_stats["sixfingerdev"]["skipped_dup"] += 1
                continue

            if normalize_text_for_match(u) in benchmark_filters:
                audit_stats["sixfingerdev"]["skipped_leak"] += 1
                continue

            sample, status, _ = format_and_tokenize(u, a, tokenizer, max_length=max_length)
            if status == "OK":
                seen_hashes.add(h)
                sample["source"] = "sixfingerdev"
                processed.append(sample)
                audit_stats["sixfingerdev"]["accepted"] += 1
            elif status == "TOO_LONG":
                audit_stats["sixfingerdev"]["skipped_length"] += 1

        # Katı Muhasebe Kontrolü: Okunan = Kabul + Retlerin Toplamı
        s_read = audit_stats["sixfingerdev"]["read"]
        s_acc = (audit_stats["sixfingerdev"]["accepted"] +
                 audit_stats["sixfingerdev"]["skipped_empty"] +
                 audit_stats["sixfingerdev"]["skipped_dup"] +
                 audit_stats["sixfingerdev"]["skipped_leak"] +
                 audit_stats["sixfingerdev"]["skipped_length"])
        assert s_read == s_acc, f"sixfingerdev sayım tutarsızlığı: {s_read} != {s_acc}"

        print(f"      ✅ sixfingerdev: {audit_stats['sixfingerdev']['accepted']} kabul edildi "
              f"(Okunan: {audit_stats['sixfingerdev']['read']}, "
              f"Boş: {audit_stats['sixfingerdev']['skipped_empty']}, "
              f"Dup: {audit_stats['sixfingerdev']['skipped_dup']}, "
              f"Sızıntı: {audit_stats['sixfingerdev']['skipped_leak']}, "
              f">512: {audit_stats['sixfingerdev']['skipped_length']})")
    except Exception as e:
        print(f"      ❌ sixfingerdev yüklenirken hata: {e}")
        raise e

    # 2. Kaynak: AlicanKiraz0/Turkish-SFT-Dataset-v1.0 (MIT)
    try:
        print(f"\n[2/3] İşleniyor: AlicanKiraz0/Turkish-SFT-Dataset-v1.0 (Hedef: ~{target_dialog_count})...")
        p2 = hf_hub_download("AlicanKiraz0/Turkish-SFT-Dataset-v1.0", "TR-part1-261025_v1-fixed.jsonl", repo_type="dataset")
        with open(p2, "r", encoding="utf-8") as f:
            for line in f:
                if audit_stats["AlicanKiraz0"]["accepted"] >= target_dialog_count:
                    break
                line = line.strip()
                if not line:
                    continue

                audit_stats["AlicanKiraz0"]["read"] += 1
                item = json.loads(line)
                u, a, sys_text = extract_qa_pair(item)

                if sys_text:
                    audit_stats["AlicanKiraz0"]["has_system_count"] += 1
                    # AlicanKiraz0'daki tek tip sentetik jenerik sistem promptu:
                    if "tüm inceliklerini kavramış" in sys_text:
                        audit_stats["AlicanKiraz0"]["generic_system_omitted"] += 1

                if not u or not a or len(u) < 4 or len(a) < 4:
                    audit_stats["AlicanKiraz0"]["skipped_empty"] += 1
                    continue

                h = hash(u.strip().lower() + "|||" + a.strip().lower())
                if h in seen_hashes:
                    audit_stats["AlicanKiraz0"]["skipped_dup"] += 1
                    continue

                if normalize_text_for_match(u) in benchmark_filters:
                    audit_stats["AlicanKiraz0"]["skipped_leak"] += 1
                    continue

                sample, status, _ = format_and_tokenize(u, a, tokenizer, max_length=max_length)
                if status == "OK":
                    seen_hashes.add(h)
                    sample["source"] = "AlicanKiraz0"
                    processed.append(sample)
                    audit_stats["AlicanKiraz0"]["accepted"] += 1
                elif status == "TOO_LONG":
                    audit_stats["AlicanKiraz0"]["skipped_length"] += 1

        # Katı Muhasebe Kontrolü: Okunan = Kabul + Retlerin Toplamı
        a_read = audit_stats["AlicanKiraz0"]["read"]
        a_acc = (audit_stats["AlicanKiraz0"]["accepted"] +
                 audit_stats["AlicanKiraz0"]["skipped_empty"] +
                 audit_stats["AlicanKiraz0"]["skipped_dup"] +
                 audit_stats["AlicanKiraz0"]["skipped_leak"] +
                 audit_stats["AlicanKiraz0"]["skipped_length"])
        assert a_read == a_acc, f"AlicanKiraz0 sayım tutarsızlığı: {a_read} != {a_acc}"

        print(f"      ✅ AlicanKiraz0: {audit_stats['AlicanKiraz0']['accepted']} kabul edildi "
              f"(Okunan: {audit_stats['AlicanKiraz0']['read']}, "
              f"Boş: {audit_stats['AlicanKiraz0']['skipped_empty']}, "
              f"Dup: {audit_stats['AlicanKiraz0']['skipped_dup']}, "
              f"Sızıntı: {audit_stats['AlicanKiraz0']['skipped_leak']}, "
              f">512: {audit_stats['AlicanKiraz0']['skipped_length']})")
    except Exception as e:
        print(f"      ❌ AlicanKiraz0 yüklenirken hata: {e}")
        raise e

    # 3. Kaynak: ogulcanaydogan/Turkish-LLM-v10-Training (Apache 2.0)
    try:
        print(f"\n[3/3] İşleniyor: ogulcanaydogan/Turkish-LLM-v10-Training (Hedef: ~{target_instruct_count})...")
        p3 = hf_hub_download("ogulcanaydogan/Turkish-LLM-v10-Training", "train.jsonl", repo_type="dataset")
        with open(p3, "r", encoding="utf-8") as f:
            for line in f:
                if audit_stats["ogulcanaydogan"]["accepted"] >= target_instruct_count:
                    break
                line = line.strip()
                if not line:
                    continue

                audit_stats["ogulcanaydogan"]["read"] += 1
                item = json.loads(line)
                u, a, _ = extract_qa_pair(item)

                if not u or not a or len(u) < 4 or len(a) < 4:
                    audit_stats["ogulcanaydogan"]["skipped_empty"] += 1
                    continue

                h = hash(u.strip().lower() + "|||" + a.strip().lower())
                if h in seen_hashes:
                    audit_stats["ogulcanaydogan"]["skipped_dup"] += 1
                    continue

                if normalize_text_for_match(u) in benchmark_filters:
                    audit_stats["ogulcanaydogan"]["skipped_leak"] += 1
                    continue

                sample, status, _ = format_and_tokenize(u, a, tokenizer, max_length=max_length)
                if status == "OK":
                    seen_hashes.add(h)
                    sample["source"] = "ogulcanaydogan"
                    processed.append(sample)
                    audit_stats["ogulcanaydogan"]["accepted"] += 1
                elif status == "TOO_LONG":
                    audit_stats["ogulcanaydogan"]["skipped_length"] += 1

        # Katı Muhasebe Kontrolü: Okunan = Kabul + Retlerin Toplamı
        o_read = audit_stats["ogulcanaydogan"]["read"]
        o_acc = (audit_stats["ogulcanaydogan"]["accepted"] +
                 audit_stats["ogulcanaydogan"]["skipped_empty"] +
                 audit_stats["ogulcanaydogan"]["skipped_dup"] +
                 audit_stats["ogulcanaydogan"]["skipped_leak"] +
                 audit_stats["ogulcanaydogan"]["skipped_length"])
        assert o_read == o_acc, f"ogulcanaydogan sayım tutarsızlığı: {o_read} != {o_acc}"

        print(f"      ✅ ogulcanaydogan: {audit_stats['ogulcanaydogan']['accepted']} kabul edildi "
              f"(Okunan: {audit_stats['ogulcanaydogan']['read']}, "
              f"Boş: {audit_stats['ogulcanaydogan']['skipped_empty']}, "
              f"Dup: {audit_stats['ogulcanaydogan']['skipped_dup']}, "
              f"Sızıntı: {audit_stats['ogulcanaydogan']['skipped_leak']}, "
              f">512: {audit_stats['ogulcanaydogan']['skipped_length']})")
    except Exception as e:
        print(f"      ❌ ogulcanaydogan yüklenirken hata: {e}")
        raise e

    print(f"\n🎉 Toplam Havuz Kabul Edilen Örnek: {len(processed):,}")
    return processed, audit_stats


def save_sft_dataset(
    samples: List[Dict[str, Any]],
    output_dir: str,
    audit_stats: Dict[str, Any],
    val_split: float = 0.10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Veriyi train ve validation kümelerine ayırır, aralarında çakışma olmadığını doğrular,
    JSONL formatında kaydeder ve kapsamlı bir metadata/istatistik raporu üretir.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    shuffled = list(samples)
    random.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    train_count = len(shuffled) - val_count

    train_data = shuffled[:train_count]
    val_data = shuffled[train_count:] if val_count > 0 else shuffled[:1]

    # Güvenlik Kontrolü: Train ve Val arasında sıfır çakışma (Zero-overlap)
    train_hashes = set(hash(s["user_text"] + "|||" + s["asst_text"]) for s in train_data)
    val_hashes = set(hash(s["user_text"] + "|||" + s["asst_text"]) for s in val_data)
    overlap = train_hashes.intersection(val_hashes)
    assert len(overlap) == 0, f"❌ ALARM: Train ve Val kümeleri arasında {len(overlap)} adet çakışan örnek bulundu!"

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    stats_path = os.path.join(output_dir, "dataset_stats.json")

    # JSONL Kayıt
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_data:
            record = {
                "tokens": s["tokens"],
                "labels": s["labels"],
                "length": s["length"],
                "prefix_len": s["prefix_len"],
                "source": s.get("source", "synthetic"),
                "user_text": s.get("user_text", ""),
                "asst_text": s.get("asst_text", "")
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_data:
            record = {
                "tokens": s["tokens"],
                "labels": s["labels"],
                "length": s["length"],
                "prefix_len": s["prefix_len"],
                "source": s.get("source", "synthetic"),
                "user_text": s.get("user_text", ""),
                "asst_text": s.get("asst_text", "")
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # İstatistikler
    lengths = [s["length"] for s in samples]
    asst_lengths = [s["asst_len"] for s in samples]
    prefix_lengths = [s["prefix_len"] for s in samples]

    # Kaynak dağılımı
    source_counts = {}
    for s in samples:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    stats = {
        "total_samples": len(samples),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "source_distribution": source_counts,
        "token_length_stats": {
            "avg_total_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_total_length": max(lengths) if lengths else 0,
            "min_total_length": min(lengths) if lengths else 0,
            "avg_prompt_tokens": sum(prefix_lengths) / len(prefix_lengths) if prefix_lengths else 0,
            "avg_assistant_tokens": sum(asst_lengths) / len(asst_lengths) if asst_lengths else 0,
            "samples_exceeding_512_filtered": sum(audit_stats[k].get("skipped_length", 0) for k in audit_stats)
        },
        "audit_details_per_source": audit_stats,
        "files": {
            "train_file": train_path,
            "val_file": val_path,
            "stats_file": stats_path
        }
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Veri seti başarıyla kaydedildi:")
    print(f"   Train : {train_path} ({len(train_data):,} örnek)")
    print(f"   Val   : {val_path} ({len(val_data):,} örnek)")
    print(f"   Stats : {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Cores-AI TR v2.0 SFT Veri Hazırlayıcı (Aşama 5B)")
    parser.add_argument("--mode", type=str, choices=["synthetic", "full"], default="full",
                        help="'synthetic' (10 test örneği) veya 'full' (HF açık veri setleri)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_tr_v2.json",
                        help="Tokenizer JSON dosya yolu")
    parser.add_argument("--benchmark", type=str, default="benchmark_questions.json",
                        help="Benchmark soruları yolu (sızıntı filtresi)")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "sft_pilot"),
                        help="Çıktı dizini (data/sft_pilot)")
    parser.add_argument("--block_size", type=int, default=512,
                        help="Maksimum token uzunluğu")
    parser.add_argument("--val_split", type=float, default=0.10,
                        help="Validation oranı (varsayılan 0.10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgelelik tohumu")
    parser.add_argument("--target_qa", type=int, default=5000,
                        help="sixfingerdev hedef örnek sayısı")
    parser.add_argument("--target_instruct", type=int, default=3500,
                        help="ogulcanaydogan hedef örnek sayısı")
    parser.add_argument("--target_dialog", type=int, default=2000,
                        help="AlicanKiraz0 hedef örnek sayısı")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Cores-AI TR v2.0 — SFT Pilot Veri Hazırlığı (Aşama 5B)")
    print("=" * 80)
    print(f"Mod         : {args.mode}")
    print(f"Tokenizer   : {args.tokenizer}")
    print(f"Benchmark   : {args.benchmark}")
    print(f"Çıktı Dizini: {args.output_dir}")
    print(f"Block Size  : {args.block_size}")

    # 1. Tokenizer yükle
    if not os.path.exists(args.tokenizer):
        print(f"❌ Hata: Tokenizer bulunamadı: {args.tokenizer}")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    print(f"✅ Tokenizer yüklendi. Vocab: {tokenizer.get_vocab_size():,} | BOS={bos_id} | EOS={eos_id} | PAD={pad_id}")

    # 2. Benchmark sızıntı filtresi yükle (Bulunamazsa hata verir ve durdurur)
    benchmark_filters = load_benchmark_leak_filters(args.benchmark)

    # 3. Veri işleme
    if args.mode == "synthetic":
        samples, audit_stats = build_synthetic_dataset(tokenizer, benchmark_filters, max_length=args.block_size)
    else:
        samples, audit_stats = build_full_dataset(
            tokenizer,
            benchmark_filters,
            max_length=args.block_size,
            target_qa_count=args.target_qa,
            target_instruct_count=args.target_instruct,
            target_dialog_count=args.target_dialog
        )

    if not samples:
        print("❌ Hata: İşlenen geçerli örnek bulunamadı!")
        sys.exit(1)

    # 4. İlk örnek üzerinde token & loss hizalama teknik doğrulamasını göster
    print_alignment_verification(samples[0], tokenizer)

    # 5. Dosyaları kaydet ve doğrulama yap
    stats = save_sft_dataset(
        samples,
        args.output_dir,
        audit_stats,
        val_split=args.val_split,
        seed=args.seed
    )

    print("\n🎉 Aşama 5B Pilot SFT Veri Seti Başarıyla Oluşturuldu ve İzolasyon Doğrulandı.")


if __name__ == "__main__":
    main()
