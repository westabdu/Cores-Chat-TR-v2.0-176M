"""
Cores-AI TR - v1 Baseline Değerlendirme Betiği
================================================
Kullanım:
    python evaluate.py \
        --out_dir out-cores-85m-test \
        --tokenizer tokenizer_tr_v2.json \
        --benchmark benchmark_questions.json \
        --output eval_v1_baseline.json \
        --device cpu

Çıktı:
    eval_v1_baseline.json  <- model çıktıları ve puanlama şablonu
    eval_v1_summary.txt    <- kategori bazlı özet istatistikler

NOT: Bu betik yalnızca model çıktılarını üretir ve kaydeder.
     Puanlama (score alanı) elle veya ayrı bir araçla doldurulacak.
"""

import os
import json
import argparse
import time
import torch
from tokenizers import Tokenizer
from model import GPTConfig, GPT

# ─── Sabit üretim parametreleri ────────────────────────────────────────────────
# Bu değerler v1 ve v2 karşılaştırmalarının tümünde DEĞİŞTİRİLMEYECEK.
EVAL_TEMPERATURE = 0.7
EVAL_TOP_K       = 50
EVAL_MAX_TOKENS  = 128
EVAL_SEED        = 42


def to_json_safe(obj):
    """
    Checkpoint'ten gelen PyTorch Tensor, numpy scalar vb. türleri
    JSON-serializable native Python tiplerine dönüştürür.
    """
    if isinstance(obj, torch.Tensor):
        # Skalar tensor ise item() ile native Python tipine çevir
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    # numpy scalar desteği (isteğe bağlı)
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def load_model(out_dir: str, device: str):
    """Checkpoint'i yükler ve model_args, iter_num, best_val_loss'u döndürür."""
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")

    print(f"\n📦 Checkpoint yükleniyor: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ─── TEKNİK GÜVENCE 1 & 2: Checkpoint alanlarını ve model yüklemesini doğrula
    required_keys = ["model", "model_args", "iter_num", "best_val_loss"]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint'te eksik alanlar: {missing}")

    # Tensor → native Python dönüşümü (JSON serialization için zorunlu)
    iter_num      = to_json_safe(checkpoint["iter_num"])
    best_val_loss = to_json_safe(checkpoint["best_val_loss"])
    model_args    = to_json_safe(checkpoint["model_args"])
    print(f"   iter_num      : {iter_num}")
    print(f"   best_val_loss : {best_val_loss:.4f}")
    print(f"   model_args    : {model_args}")

    # Gradient checkpointing çıkarım sırasında kapatılacak
    if "use_gradient_checkpointing" in model_args:
        model_args["use_gradient_checkpointing"] = False

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"   ✅ Model yüklendi. Parametre sayısı: {model.get_num_params():,}")

    return model, model_args, iter_num, best_val_loss


def load_tokenizer(tokenizer_path: str):
    """Tokenizer'ı yükler ve özel token kimliklerini doğrular."""
    print(f"\n📖 Tokenizer yükleniyor: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    bos_id     = tokenizer.token_to_id("[BOS]")
    eos_id     = tokenizer.token_to_id("[EOS]")
    pad_id     = tokenizer.token_to_id("[PAD]")

    # ─── TEKNİK GÜVENCE 6: Özel tokenların gerçek ID'lere karşılık geldiğini doğrula
    assert bos_id is not None, "[BOS] tokenı bulunamadı!"
    assert eos_id is not None, "[EOS] tokenı bulunamadı!"
    encoded_bos = tokenizer.encode("[BOS]").ids
    print(f"   Vocab: {vocab_size} | [BOS]={bos_id} | [EOS]={eos_id} | [PAD]={pad_id}")
    print(f"   '[BOS]' encode test: {encoded_bos}  (beklenen: [{bos_id}])")

    return tokenizer, bos_id, eos_id, pad_id


def generate(model, tokenizer, prompt: str, bos_id: int, eos_id: int,
             device: str, temperature: float, top_k: int, max_new_tokens: int) -> str:
    """
    Prompt'tan metin üretir.

    ─── TEKNİK GÜVENCE 5: EOS durma kontrolü ────────────────────────────────
    model.generate() çağrısı EOS tokenını durma koşulu olarak kullanmıyor.
    Bu betik üretilen token listesinde eos_id'yi göründüğü anda keser.
    sample_tr.py'nin generate() döngüsü kontrol edilmeli; eğer orada da
    eos_id'de durma yoksa hem bu betik hem train_sft.py için not düşülecek.
    """
    encoded    = tokenizer.encode(prompt)
    start_ids  = encoded.ids

    # BOS ekle (sample_tr.py ile tutarlı)
    start_ids = [bos_id] + start_ids

    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        device_type = "cuda" if "cuda" in device else "cpu"
        ptdtype = torch.float16 if device_type == "cuda" else torch.float32
        ctx = (torch.amp.autocast(device_type=device_type, dtype=ptdtype)
               if device_type == "cuda" else torch.no_grad())
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    generated_ids = y[0].tolist()

    # EOS'ta kes (TEKNİK GÜVENCE 5)
    eos_found = False
    if eos_id in generated_ids:
        eos_pos = generated_ids.index(eos_id)
        generated_ids = generated_ids[: eos_pos + 1]
        eos_found = True

    text = tokenizer.decode(generated_ids)
    return text, eos_found


def main():
    parser = argparse.ArgumentParser(description="Cores-AI TR Baseline Değerlendirme")
    parser.add_argument("--out_dir",    type=str, default="out-cores-85m-test")
    parser.add_argument("--tokenizer",  type=str, default="tokenizer_tr_v2.json")
    parser.add_argument("--benchmark",  type=str, default="benchmark_questions.json")
    parser.add_argument("--output",     type=str, default="eval_v1_baseline.json")
    parser.add_argument("--device",     type=str, default="cpu",
                        help="cuda veya cpu. GPU yoksa cpu kullanın.")
    parser.add_argument("--categories", type=str, default=None,
                        help="Virgülle ayrılmış kategori filtreleri. Örn: A,B,C")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Kaç soru çalıştırılacak (test için). Tüm sorular için boş bırak.")
    args = parser.parse_args()

    # Seed sabitle
    torch.manual_seed(EVAL_SEED)
    if "cuda" in args.device:
        torch.cuda.manual_seed(EVAL_SEED)

    # ─── Model ve tokenizer yükle
    model, model_args, iter_num, best_val_loss = load_model(args.out_dir, args.device)
    tokenizer, bos_id, eos_id, pad_id         = load_tokenizer(args.tokenizer)

    # ─── Benchmark yükle
    print(f"\n📋 Benchmark yükleniyor: {args.benchmark}")
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    questions = benchmark["questions"]

    # Kategori filtresi
    if args.categories:
        cats = set(args.categories.split(","))
        questions = [q for q in questions if q["category"] in cats]
        print(f"   Kategori filtresi: {cats} → {len(questions)} soru")

    # Limit
    if args.limit:
        questions = questions[:args.limit]
        print(f"   Limit uygulandı: {args.limit} soru")

    print(f"   Toplam çalıştırılacak: {len(questions)} soru")
    print(f"\n🔧 Sabit üretim ayarları:")
    print(f"   temperature={EVAL_TEMPERATURE}, top_k={EVAL_TOP_K}, max_new_tokens={EVAL_MAX_TOKENS}")

    # ─── Üretim döngüsü
    results = []
    eos_hit_count = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"{'🚀 DEĞERLENDİRME BAŞLIYOR':^60}")
    print(f"{'='*60}")

    for i, q in enumerate(questions):
        prompt        = q["prompt"]
        expected      = q["expected_behavior"]
        qid           = q["id"]
        category      = q["category"]

        output_text, eos_found = generate(
            model, tokenizer, prompt,
            bos_id, eos_id, args.device,
            EVAL_TEMPERATURE, EVAL_TOP_K, EVAL_MAX_TOKENS
        )

        if eos_found:
            eos_hit_count += 1

        result = {
            "id":               qid,
            "category":         category,
            "prompt":           prompt,
            "expected_behavior": expected,
            "output":           output_text,
            "eos_found":        eos_found,
            "score":            None,   # Manuel doldurulacak
            "notes":            ""      # Opsiyonel yorum alanı
        }
        results.append(result)

        # İlerleme
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"[{i+1:>3}/{len(questions)}] ({elapsed:.1f}s) "
                  f"{qid}: {prompt[:50]!r}")
            print(f"          → {output_text[:80]!r}{'...' if len(output_text)>80 else ''}")
            print()

    elapsed_total = time.time() - t_start

    # ─── Sonuçları kaydet
    output_data = {
        "meta": {
            "model_checkpoint":  os.path.join(args.out_dir, "ckpt.pt"),
            "iter_num":          iter_num,
            "best_val_loss":     best_val_loss,
            "model_args":        model_args,
            "tokenizer":         args.tokenizer,
            "eval_params": {
                "temperature":   EVAL_TEMPERATURE,
                "top_k":         EVAL_TOP_K,
                "max_new_tokens": EVAL_MAX_TOKENS,
                "seed":          EVAL_SEED,
                "device":        args.device
            },
            "total_questions":   len(results),
            "eos_hit_count":     eos_hit_count,
            "eos_hit_rate":      f"{eos_hit_count/len(results)*100:.1f}%" if results else "N/A",
            "elapsed_seconds":   round(elapsed_total, 1),
            "scoring_guide": benchmark["meta"]["scoring"]
        },
        "results": results
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(output_data), f, ensure_ascii=False, indent=2)

    # ─── Özet rapor
    summary_path = args.output.replace(".json", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CORES-AI TR v1 BASELINE DEĞERLENDİRME ÖZETİ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint   : {args.out_dir}/ckpt.pt\n")
        f.write(f"iter_num     : {iter_num}\n")
        f.write(f"best_val_loss: {best_val_loss:.4f}\n\n")
        f.write(f"Toplam soru  : {len(results)}\n")
        f.write(f"EOS vurma    : {eos_hit_count}/{len(results)} ({eos_hit_count/len(results)*100:.1f}%)\n")
        f.write(f"Süre         : {elapsed_total:.1f}s\n\n")
        f.write("PUANLAMA ŞABLONU (manuel doldurun):\n")
        f.write("-" * 40 + "\n")
        f.write("Kategori | Puan Ortalaması | Notlar\n")
        for cat_id, cat_info in benchmark["categories"].items():
            cat_results = [r for r in results if r["category"] == cat_id]
            f.write(f"  {cat_id} - {cat_info['name']:<35}: {len(cat_results)} soru → [ ]\n")
        f.write("\n")
        f.write("PUANLAMA REHBERI:\n")
        for score, desc in benchmark["meta"]["scoring"].items():
            f.write(f"  {score}: {desc}\n")

    print(f"\n{'='*60}")
    print(f"✅ TAMAMLANDI")
    print(f"{'='*60}")
    print(f"Çıktı dosyası  : {args.output}")
    print(f"Özet raporu    : {summary_path}")
    print(f"EOS vurma oranı: {eos_hit_count}/{len(results)} ({eos_hit_count/len(results)*100:.1f}%)")
    print(f"Toplam süre    : {elapsed_total:.1f}s")
    print(f"\n⚠️  'score' alanları boş bırakıldı — lütfen her çıktıyı okuyarak doldurun.")
    print(f"    Dosyayı JSON editörünüzde açın veya manuel düzenleyin: {args.output}")


if __name__ == "__main__":
    main()
