"""
Cores-AI TR v2.0 — SFT Karsilastirmali Degerlendirme Betigi (evaluate_sft.py)
==============================================================================

Kullanim:
    .venv/Scripts/python evaluate_sft.py \
        --ckpt_path TESTLER/temp_sft_pilot/ckpt_best.pt \
        --benchmark TESTLER/benchmark_questions.json \
        --output TESTLER/eval_v2_sft.json \
        --device cpu

Guvenceler:
  - evaluate.py dosyasina DOKUNULMAZ.
  - eval_v1_baseline.json dosyasina DOKUNULMAZ.
  - benchmark_questions.json dosyasina DOKUNULMAZ.
  - out-cores-85m-test/ckpt.pt SHA-256 degerlendirme oncesi ve sonrasi dogrulanir.
  - Cikti yalnizca --output ile belirtilen dosyaya ve ozet txt dosyasina yazilir.

Sabit uretim parametreleri (evaluate.py ile birebir ayni):
  EVAL_TEMPERATURE = 0.7
  EVAL_TOP_K       = 50
  EVAL_MAX_TOKENS  = 128
  EVAL_SEED        = 42
"""

import os
import sys
import json
import time
import hashlib
import argparse
from typing import Dict, Any, List, Tuple

import torch
from tokenizers import Tokenizer
from model import GPTConfig, GPT

# Sabit uretim parametreleri - evaluate.py ile BIREBIR AYNI
EVAL_TEMPERATURE = 0.7
EVAL_TOP_K       = 50
EVAL_MAX_TOKENS  = 128
EVAL_SEED        = 42

# Windows UTF-8 cikti destegi
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def calculate_sha256(filepath: str) -> str:
    """Dosyanin SHA-256 hash degerini dondurur."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def to_json_safe(obj: Any) -> Any:
    """Tensor, numpy vb. turleri JSON serialize edilebilir tiplere donusturur."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
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


def load_sft_model(ckpt_path: str, device: str = "cpu") -> Tuple[GPT, Dict[str, Any], int, float]:
    """SFT checkpoint dosyasini yukler ve dogrular."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint bulunamadi: {ckpt_path}")

    print(f"\n📦 SFT Checkpoint yukleniyor: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    required_keys = ["model", "model_args"]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint icinde eksik alanlar: {missing}")

    iter_num = to_json_safe(checkpoint.get("iter_num", -1))
    val_loss = to_json_safe(checkpoint.get("val_loss", checkpoint.get("best_val_loss", -1.0)))
    model_args = to_json_safe(checkpoint["model_args"])

    print(f"   iter_num : {iter_num}")
    print(f"   val_loss : {val_loss}")
    print(f"   model_args: {model_args}")

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

    param_count = model.get_num_params()
    print(f"   ✅ Model yuklendi. Parametre sayisi: {param_count:,}")
    return model, model_args, iter_num, val_loss


def load_tokenizer(tokenizer_path: str) -> Tuple[Tokenizer, int, int, int]:
    """Tokenizer ve ozel token ID'lerini dogrulayarak yukler."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer bulunamadi: {tokenizer_path}")

    print(f"\n📖 Tokenizer yukleniyor: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    assert bos_id is not None, "[BOS] token ID bulunamadi!"
    assert eos_id is not None, "[EOS] token ID bulunamadi!"

    print(f"   Vocab: {tokenizer.get_vocab_size()} | [BOS]={bos_id} | [EOS]={eos_id} | [PAD]={pad_id}")
    return tokenizer, bos_id, eos_id, pad_id


def generate_sft_response(
    model: GPT,
    tokenizer: Tokenizer,
    raw_prompt: str,
    bos_id: int,
    eos_id: int,
    device: str = "cpu",
    temperature: float = EVAL_TEMPERATURE,
    top_k: int = EVAL_TOP_K,
    max_new_tokens: int = EVAL_MAX_TOKENS,
    use_sft_template: bool = True
) -> Tuple[str, str, bool, int]:
    """
    SFT formatli prompt uretir.
    Prompt sablonu:
      use_sft_template=True -> [BOS]Kullanici: {raw_prompt}\nAsistan:
      use_sft_template=False -> [BOS]{raw_prompt}
    
    Donus: (tam_metin, asistan_yaniti, eos_bulundu_mu, uretilen_token_sayisi)
    """
    if use_sft_template:
        formatted_prompt = f"Kullanıcı: {raw_prompt.strip()}\nAsistan:"
    else:
        formatted_prompt = raw_prompt.strip()

    encoded = tokenizer.encode(formatted_prompt)
    prompt_ids = [bos_id] + encoded.ids
    prompt_len = len(prompt_ids)

    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        device_type = "cuda" if "cuda" in device else "cpu"
        ptdtype = torch.float16 if device_type == "cuda" else torch.float32
        ctx = (torch.amp.autocast(device_type=device_type, dtype=ptdtype)
               if device_type == "cuda" else torch.no_grad())
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    all_ids = y[0].tolist()
    generated_new_ids = all_ids[prompt_len:]

    eos_found = False
    if eos_id in generated_new_ids:
        eos_pos = generated_new_ids.index(eos_id)
        cut_new_ids = generated_new_ids[:eos_pos]
        eos_found = True
    else:
        cut_new_ids = generated_new_ids

    tokens_generated = len(cut_new_ids)
    assistant_response = tokenizer.decode(cut_new_ids).strip()

    # Tam metin (baseline ile birebir karsilastirilabilir cikti)
    if eos_found:
        full_ids = prompt_ids + cut_new_ids + [eos_id]
    else:
        full_ids = prompt_ids + cut_new_ids
    full_text = tokenizer.decode(full_ids)

    return full_text, assistant_response, eos_found, tokens_generated


def main():
    parser = argparse.ArgumentParser(description="Cores-AI TR v2 SFT Degerlendirme")
    parser.add_argument("--ckpt_path", type=str, default="TESTLER/temp_sft_pilot/ckpt_best.pt",
                        help="SFT model checkpoint yolu")
    parser.add_argument("--baseline_ckpt", type=str, default="out-cores-85m-test/ckpt.pt",
                        help="Orijinal baseline checkpoint yolu (SHA dogrulamasi icin)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_tr_v2.json")
    parser.add_argument("--benchmark", type=str, default="TESTLER/benchmark_questions.json")
    parser.add_argument("--output", type=str, default="TESTLER/eval_v2_sft.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_sft_template", action="store_true",
                        help="SFT diyalog sablonunu kapatip ham prompt kullanir")
    args = parser.parse_args()

    use_sft_template = not args.no_sft_template

    # 1. Orijinal checkpoint SHA-256 ilk kontrolu
    if os.path.exists(args.baseline_ckpt):
        sha_initial = calculate_sha256(args.baseline_ckpt)
        print(f"\n🔒 Orijinal baseline checkpoint SHA-256 (Baslangic):")
        print(f"   {args.baseline_ckpt} -> {sha_initial}")
    else:
        print(f"⚠️ Uyari: Baseline checkpoint bulunamadi: {args.baseline_ckpt}")
        sha_initial = None

    # 2. Seed sabitleme
    torch.manual_seed(EVAL_SEED)
    if "cuda" in args.device:
        torch.cuda.manual_seed(EVAL_SEED)

    # 3. Model ve Tokenizer yukleme
    model, model_args, iter_num, val_loss = load_sft_model(args.ckpt_path, args.device)
    tokenizer, bos_id, eos_id, pad_id = load_tokenizer(args.tokenizer)

    # 4. Benchmark sorularini yukleme
    if not os.path.exists(args.benchmark):
        raise FileNotFoundError(f"Benchmark dosyasi bulunamadi: {args.benchmark}")

    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    questions = benchmark["questions"]
    if args.categories:
        cats = set(args.categories.split(","))
        questions = [q for q in questions if q["category"] in cats]
        print(f"   Kategori filtresi: {cats} -> {len(questions)} soru")

    if args.limit:
        questions = questions[:args.limit]
        print(f"   Limit uygulandi: {args.limit} soru")

    total_q = len(questions)
    print(f"\n📋 Toplam calistirilacak soru sayisi: {total_q}")
    print(f"🔧 Uretim parametreleri: temp={EVAL_TEMPERATURE}, top_k={EVAL_TOP_K}, max_tokens={EVAL_MAX_TOKENS}, seed={EVAL_SEED}")
    print(f"🎭 Prompt sablonu: {'[BOS]Kullanici: {soru}\\nAsistan:' if use_sft_template else '[BOS]{soru}'}")

    # 5. Uretim dongusu
    print(f"\n{'='*70}")
    print(f"{'🚀 SFT KARŞILAŞTIRMALI DEĞERLENDİRME BAŞLIYOR':^70}")
    print(f"{'='*70}")

    results = []
    eos_hit_count = 0
    total_tokens_gen = 0
    cat_stats = {cat: {"total": 0, "eos": 0, "tokens": 0} for cat in benchmark.get("categories", {})}

    t_start = time.time()
    errors_count = 0

    for i, q in enumerate(questions):
        qid = q["id"]
        cat = q["category"]
        prompt = q["prompt"]
        expected = q.get("expected_behavior", "")

        try:
            full_text, asst_resp, eos_found, tokens_gen = generate_sft_response(
                model=model,
                tokenizer=tokenizer,
                raw_prompt=prompt,
                bos_id=bos_id,
                eos_id=eos_id,
                device=args.device,
                temperature=EVAL_TEMPERATURE,
                top_k=EVAL_TOP_K,
                max_new_tokens=EVAL_MAX_TOKENS,
                use_sft_template=use_sft_template
            )

            if eos_found:
                eos_hit_count += 1
            total_tokens_gen += tokens_gen

            if cat in cat_stats:
                cat_stats[cat]["total"] += 1
                if eos_found:
                    cat_stats[cat]["eos"] += 1
                cat_stats[cat]["tokens"] += tokens_gen

            result_item = {
                "id": qid,
                "category": cat,
                "prompt": prompt,
                "expected_behavior": expected,
                "output": full_text,
                "assistant_response": asst_resp,
                "eos_found": eos_found,
                "generated_tokens": tokens_gen,
                "score": None,
                "notes": ""
            }
            results.append(result_item)

            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total_q:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remain = (total_q - (i + 1)) / rate if rate > 0 else 0
                eos_pct = (eos_hit_count / (i + 1)) * 100
                print(f"[{i+1:>3}/{total_q}] ({elapsed:.1f}s | ~{remain:.0f}s kaldi) "
                      f"EOS: {eos_pct:.1f}% | {qid}: {prompt[:35]!r}")
                print(f"          → Asistan: {asst_resp[:70]!r}{'...' if len(asst_resp)>70 else ''}")

        except Exception as e:
            errors_count += 1
            print(f"❌ HATA [{qid}]: {e}")
            results.append({
                "id": qid,
                "category": cat,
                "prompt": prompt,
                "expected_behavior": expected,
                "output": f"ERROR: {e}",
                "assistant_response": "",
                "eos_found": False,
                "generated_tokens": 0,
                "score": 1,
                "notes": f"Generation error: {e}"
            })

    total_elapsed = time.time() - t_start
    avg_tokens = total_tokens_gen / len(results) if results else 0
    eos_rate = (eos_hit_count / len(results)) * 100 if results else 0

    # 6. Orijinal checkpoint SHA-256 son kontrolu
    if os.path.exists(args.baseline_ckpt):
        sha_final = calculate_sha256(args.baseline_ckpt)
        sha_match = (sha_initial == sha_final)
        print(f"\n🔒 Orijinal baseline checkpoint SHA-256 (Bitis):")
        print(f"   {args.baseline_ckpt} -> {sha_final}")
        print(f"   Bütünlük korundu mu? {'✅ EVET' if sha_match else '❌ HAYIR - DEGISMIS!'}")
        assert sha_match, "KRITIK GUVENLIK IHLALI: Orijinal checkpoint SHA degeri degisti!"
    else:
        sha_final = None
        sha_match = None

    # 7. JSON Çıktısını Kaydet
    output_data = {
        "meta": {
            "model_version": "v2.0-SFT-Pilot",
            "model_checkpoint": args.ckpt_path,
            "iter_num": iter_num,
            "val_loss": val_loss,
            "baseline_checkpoint": args.baseline_ckpt,
            "baseline_sha256": sha_final,
            "baseline_sha256_verified": sha_match,
            "model_args": model_args,
            "tokenizer": args.tokenizer,
            "eval_params": {
                "temperature": EVAL_TEMPERATURE,
                "top_k": EVAL_TOP_K,
                "max_new_tokens": EVAL_MAX_TOKENS,
                "seed": EVAL_SEED,
                "device": args.device,
                "use_sft_template": use_sft_template,
                "prompt_template": "Kullanıcı: {prompt}\\nAsistan:" if use_sft_template else "raw"
            },
            "total_questions": len(results),
            "completed_questions": len(results) - errors_count,
            "error_count": errors_count,
            "eos_hit_count": eos_hit_count,
            "eos_hit_rate": f"{eos_rate:.1f}%",
            "avg_generated_tokens": round(avg_tokens, 1),
            "elapsed_seconds": round(total_elapsed, 1),
            "scoring_guide": benchmark.get("meta", {}).get("scoring", {})
        },
        "results": results
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(output_data), f, ensure_ascii=False, indent=2)

    # 8. Ozet Metin Raporunu Kaydet
    summary_path = args.output.replace(".json", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CORES-AI TR v2 SFT DEGERLENDIRME OZET RAPORU\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint           : {args.ckpt_path}\n")
        f.write(f"Iterasyon            : {iter_num}\n")
        f.write(f"Val Loss             : {val_loss}\n")
        f.write(f"Baseline Checkpoint  : {args.baseline_ckpt}\n")
        f.write(f"Baseline SHA-256     : {sha_final} ({'GÜVENLİ/DEĞİŞMEDİ' if sha_match else 'HATA'})\n")
        f.write(f"Toplam Soru          : {len(results)}\n")
        f.write(f"Hata Sayisi          : {errors_count}\n")
        f.write(f"EOS Vurma Orani      : {eos_hit_count}/{len(results)} ({eos_rate:.1f}%)\n")
        f.write(f"Ort. Uretilen Token  : {avg_tokens:.1f}\n")
        f.write(f"Toplam Sure          : {total_elapsed:.1f}s ({total_elapsed/60:.1f} dakika)\n\n")
        f.write("KATEGORI BAZLI DAGILIM:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Kat':<4} | {'Kategori Adi':<32} | {'Soru':<5} | {'EOS %':<8} | {'Ort. Tok':<8}\n")
        f.write("-" * 70 + "\n")
        for cat_id, cat_info in benchmark.get("categories", {}).items():
            st = cat_stats.get(cat_id, {"total": 0, "eos": 0, "tokens": 0})
            cnt = st["total"]
            pct = (st["eos"] / cnt * 100) if cnt > 0 else 0
            avg_t = (st["tokens"] / cnt) if cnt > 0 else 0
            f.write(f"{cat_id:<4} | {cat_info['name']:<32} | {cnt:<5} | %{pct:<7.1f} | {avg_t:<8.1f}\n")
        f.write("-" * 70 + "\n")

    print(f"\n{'='*70}")
    print(f"✅ DEĞERLENDİRME BAŞARIYLA TAMAMLANDI")
    print(f"{'='*70}")
    print(f"JSON Cikti       : {args.output}")
    print(f"Ozet Rapor       : {summary_path}")
    print(f"EOS Vurma Orani  : {eos_hit_count}/{len(results)} ({eos_rate:.1f}%)")
    print(f"Ortalama Token   : {avg_tokens:.1f}")
    print(f"Toplam Sure      : {total_elapsed:.1f}s (~{total_elapsed/60:.1f} dk)")
    print(f"Baseline SHA-256 : {'✅ KORUNDU' if sha_match else '❌ DEĞİŞTİ'}")


if __name__ == "__main__":
    main()
