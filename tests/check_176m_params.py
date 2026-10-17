"""
Cores-AI TR 176M — Mimari ve Parametre Doğrulama Testi
======================================================
Bu test:
1. configs/train_cores_176m.py yapılandırma dosyasını okur.
2. 176M mimarisini temsil eden GPTConfig nesnesini ve GPT modelini oluşturur (eğitim başlatmaz).
3. Toplam ve konum embeddingleri hariç parametre sayılarını hesaplar.
4. Katman, kafa, embedding boyutu, sözlük boyutu ve ağırlık paylaşımını doğrular.
5. Parametre sayısının 175M - 177M aralığında (hedef: ~176.021.248) olduğunu teyit eder.
6. Tokenizer varlığını kontrol eder ancak test için zorunlu tutmaz.
"""

import os
import sys

# Windows UTF-8 konsol desteği
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Proje kök dizinini sys.path'e ekle
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    import torch
except ImportError:
    print("❌ HATA: 'torch' kütüphanesi bulunamadı! Lütfen PyTorch kurulu ortamı kullanın.")
    sys.exit(1)

from model import GPTConfig, GPT


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı: {config_path}")
    cfg = {}
    with open(config_path, "r", encoding="utf-8") as f:
        exec(f.read(), cfg)
    return cfg


def main():
    print("=" * 80)
    print("🧪 CORES-AI TR 176M — MİMARİ VE PARAMETRE DOĞRULAMA TESTİ")
    print("=" * 80)

    config_path = os.path.join(root_dir, "configs", "train_cores_176m.py")
    print(f"📂 Yapılandırma dosyası: {config_path}")

    try:
        cfg = load_config(config_path)
        print("   ✅ Yapılandırma başarıyla yüklendi.")
    except Exception as e:
        print(f"   ❌ Yapılandırma okuma hatası: {e}")
        sys.exit(1)

    # Hedef mimari parametrelerini çek
    n_layer = cfg.get("n_layer")
    n_head = cfg.get("n_head")
    n_embd = cfg.get("n_embd")
    block_size = cfg.get("block_size")
    vocab_size = cfg.get("vocab_size")
    bias = cfg.get("bias")
    use_gradient_checkpointing = cfg.get("use_gradient_checkpointing")
    batch_size = cfg.get("batch_size")
    grad_accum = cfg.get("gradient_accumulation_steps")
    compile_flag = cfg.get("compile")

    print("\n🔍 Yapılandırma Değerleri Kontrolü:")
    print(f"   - n_layer                    : {n_layer} (Beklenen: 18)")
    print(f"   - n_head                     : {n_head} (Beklenen: 13)")
    print(f"   - n_embd                     : {n_embd} (Beklenen: 832)")
    print(f"   - block_size (bağlam)        : {block_size} (Beklenen: 512)")
    print(f"   - vocab_size (sözlük)        : {vocab_size} (Beklenen: 31104)")
    print(f"   - bias                       : {bias} (Beklenen: True)")
    print(f"   - use_gradient_checkpointing : {use_gradient_checkpointing} (Beklenen: True)")
    print(f"   - batch_size (mikro)         : {batch_size} (Beklenen: 1)")
    print(f"   - gradient_accumulation_steps: {grad_accum} (Beklenen: 64)")
    print(f"   - compile (PyTorch 2)        : {compile_flag} (Beklenen: False)")

    assert n_layer == 18, f"Hatalı n_layer: {n_layer}"
    assert n_head == 13, f"Hatalı n_head: {n_head}"
    assert n_embd == 832, f"Hatalı n_embd: {n_embd}"
    assert block_size == 512, f"Hatalı block_size: {block_size}"
    assert vocab_size == 31104, f"Hatalı vocab_size: {vocab_size}"
    assert bias is True, f"Hatalı bias: {bias}"
    assert use_gradient_checkpointing is True, "use_gradient_checkpointing True olmalıdır!"
    assert batch_size == 1, f"batch_size 1 olmalıdır, alınan: {batch_size}"
    assert grad_accum == 64, f"gradient_accumulation_steps 64 olmalıdır, alınan: {grad_accum}"
    assert compile_flag is False, "Windows üzerinde stabilite için compile False olmalıdır!"

    head_dim = n_embd // n_head
    assert n_embd % n_head == 0, f"n_embd ({n_embd}) n_head ({n_head})'e tam bölünmüyor!"
    print(f"   - Kafa boyutu (head dimension): {head_dim} (832 / 13 = 64)")

    # Modeli CPU üzerinde başlat (eğitim yok, yalnızca tensör tahsisi)
    print("\n🧠 Model Nesnesi Başlatılıyor (CPU üzerinde, eğitime başlanmaz)...")
    gptconf = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing
    )

    with torch.no_grad():
        model = GPT(gptconf)

    # Parametre hesaplamaları
    total_unique_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.get_num_params(non_embedding=True)
    pos_emb_params = model.transformer.wpe.weight.numel()
    token_emb_params = model.transformer.wte.weight.numel()

    # Ağırlık paylaşımı kontrolü (wte vs lm_head)
    wte_ptr = model.transformer.wte.weight.data_ptr()
    lm_head_ptr = model.lm_head.weight.data_ptr()
    is_weight_tied = (model.transformer.wte.weight is model.lm_head.weight) and (wte_ptr == lm_head_ptr)

    print("\n" + "=" * 80)
    print("📊 PARAMETRE SAYIM VE MİMARİ DOĞRULAMA SONUÇLARI")
    print("=" * 80)
    print(f"   🔹 Toplam Benzersiz Parametre Sayısı       : {total_unique_params:,} (~{total_unique_params/1e6:.2f}M)")
    print(f"   🔹 Konum Embeddingleri Hariç Parametre     : {non_emb_params:,} (~{non_emb_params/1e6:.2f}M)")
    print(f"   🔹 Token Embedding Boyutu (wte)            : {token_emb_params:,} (vocab_size * n_embd = {vocab_size} * {n_embd})")
    print(f"   🔹 Konum Embedding Boyutu (wpe)            : {pos_emb_params:,} (block_size * n_embd = {block_size} * {n_embd})")
    print(f"   🔹 Katman Sayısı (n_layer)                 : {n_layer}")
    print(f"   🔹 Dikkat Kafası Sayısı (n_head)           : {n_head}")
    print(f"   🔹 Embedding Boyutu (n_embd)               : {n_embd}")
    print(f"   🔹 Sözlük Boyutu (vocab_size)              : {vocab_size}")
    print(f"   🔹 Ağırlık Paylaşımı (Weight Tying)        : {'✅ AKTİF (wte == lm_head)' if is_weight_tied else '❌ PASİF'}")

    # Doğrulama kriterleri
    TARGET_EXACT = 176_021_248
    RANGE_MIN = 175_000_000
    RANGE_MAX = 177_000_000

    in_range = (RANGE_MIN <= total_unique_params <= RANGE_MAX)
    is_exact = (total_unique_params == TARGET_EXACT)

    print("\n🎯 Kriter Kontrolleri:")
    print(f"   - Parametre 175M - 177M aralığında mı?      : {'✅ EVET' if in_range else '❌ HAYIR'}")
    print(f"   - Hedef 176.021.248 ile birebir eşleşme     : {'✅ BİREBİR EŞLEŞTİ' if is_exact else f'Fark: {total_unique_params - TARGET_EXACT:+d}'}")
    print(f"   - Ağırlık paylaşımı korundu mu?            : {'✅ EVET' if is_weight_tied else '❌ HAYIR'}")

    assert is_weight_tied, "Ağırlık paylaşımı (weight tying) aktif değil!"
    assert in_range, f"Toplam parametre ({total_unique_params}) hedef aralıkta ({RANGE_MIN} - {RANGE_MAX}) değil!"

    # Tokenizer ve veri meta durumu kontrolü
    print("\n📦 Tokenizer ve Veri Meta Bilgisi Durumu:")
    tokenizer_candidates = [
        os.path.join(root_dir, "tokenizer_tr_176m.json"),
        os.path.join(root_dir, "tokenizer_tr_v2.json")
    ]
    found_tokenizer = None
    for cand in tokenizer_candidates:
        if os.path.exists(cand):
            found_tokenizer = cand
            break

    if found_tokenizer:
        print(f"   ℹ️  Bulunan tokenizer dosyası: {os.path.relpath(found_tokenizer, root_dir)}")
    else:
        print("   ⚠️  Bilgi: Henüz 176M'e özel tokenizer dosyası bulunamadı.")

    print("   📢 NOT: Mimari doğrulama için tokenizer dosyası ZORUNLU DEĞİLDİR.")
    print("   📢 Ancak gerçek eğitime başlamadan önce 31.104 kelimelik tokenizer ve veri seti")
    print("   📢 meta bilgilerinin (meta.pkl) hazırlanması ve doğrulanması gerekmektedir.")

    print("\n" + "=" * 80)
    print("🎉 SONUÇ: CORES-AI TR 176M MİMARİ VE PARAMETRE DOĞRULAMASI BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
