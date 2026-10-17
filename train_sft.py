"""
Cores-AI TR v2.0 — SFT Eğitim Betiği (train_sft.py)
Aşama 5: Supervised Fine-Tuning Pilot & Tam Eğitim Motoru

Bu betik:
1. Mevcut v1 base model checkpoint'ini (out-cores-85m-test/ckpt.pt) salt-okunur olarak yükler.
2. SFT için data_sft.py DataLoader'ını kullanır (dinamik padding, -1 loss maskeleme).
3. Yalnızca Asistan yanıtı ve [EOS] tokenı üzerinden gradyan hesaplar.
4. Cosine learning rate schedule (düşük LR: 2e-5) ve AdamW optimizer kullanır.
5. Sadece validation loss iyileştiğinde checkpoint kaydeder (always_save_checkpoint=False).
6. Orijinal checkpoint'i asla ezmez; çıktıları belirtilen out_dir altına kaydeder.
"""

import os
import sys
import time
import math
import hashlib
import argparse
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F

# Windows konsolunda UTF-8 çıktı güvencesi
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from model import GPT, GPTConfig
from data_sft import create_sft_dataloader, TOKEN_PAD_ID, TOKEN_IGNORE_TARGET


def calculate_sha256(filepath: str) -> str:
    """Dosyanın SHA256 özetini hesaplar."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_lr(it: int, max_iters: int, warmup_iters: int, learning_rate: float, min_lr: float) -> float:
    """Cosine learning rate scheduler with warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataloader, device: str, max_batches: int = 5) -> float:
    """DataLoader üzerinden ortalama loss hesaplar."""
    model.eval()
    losses = []
    for i, (X, Y) in enumerate(dataloader):
        if i >= max_batches:
            break
        X, Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        if torch.isfinite(loss):
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses)) if losses else float("nan")


def train_sft(
    init_ckpt: str,
    out_dir: str,
    data_dir: str,
    max_iters: int = 5,
    eval_interval: int = 2,
    learning_rate: float = 2e-5,
    min_lr: float = 2e-6,
    warmup_iters: int = 1,
    weight_decay: float = 1e-2,
    batch_size: int = 2,
    grad_accum_steps: int = 1,
    device: str = "cpu"
) -> Dict[str, Any]:
    print("=" * 80)
    print("🚀 CORES-AI TR v2.0 — SFT EĞİTİM MOTORU (train_sft.py)")
    print("=" * 80)
    print(f"Başlangıç Checkpoint : {init_ckpt}")
    print(f"Çıktı Dizini         : {out_dir}")
    print(f"Veri Dizini          : {data_dir}")
    print(f"Cihaz                : {device.upper()}")
    print(f"Maksimum İterasyon   : {max_iters}")
    print(f"Learning Rate        : {learning_rate} (Min LR: {min_lr}, Warmup: {warmup_iters})")
    print(f"Batch Boyutu         : {batch_size} (Grad Accum: {grad_accum_steps})")

    # 1. Güvenlik Kilidi: Orijinal checkpoint SHA256 özetini al
    if not os.path.exists(init_ckpt):
        raise FileNotFoundError(f"Başlangıç checkpoint'i bulunamadı: {init_ckpt}")

    sha_before = calculate_sha256(init_ckpt)
    print(f"\n🔒 Orijinal Checkpoint SHA256 (Eğitim Öncesi): {sha_before}")

    # 2. Çıktı dizinini hazırla
    os.makedirs(out_dir, exist_ok=True)

    # 3. Model ve ağırlıkları yükle
    print(f"\n📦 Checkpoint yükleniyor: {init_ckpt}")
    checkpoint = torch.load(init_ckpt, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]

    # Eğitimde gradient checkpointing bellek optimizasyonu için korunabilir veya kapatılabilir
    if "use_gradient_checkpointing" in model_args:
        model_args["use_gradient_checkpointing"] = False

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Parametre ağırlıklarını yükle
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    print(f"   ✅ Model hazırlandı. Toplam Parametre: {model.get_num_params():,}")

    # 4. DataLoader'ları hazırla
    train_jsonl = os.path.join(data_dir, "train.jsonl")
    val_jsonl = os.path.join(data_dir, "val.jsonl")

    train_loader = create_sft_dataloader(
        jsonl_path=train_jsonl,
        batch_size=batch_size,
        shuffle=True,
        pad_id=TOKEN_PAD_ID,
        pad_target=TOKEN_IGNORE_TARGET,
        block_size=model_args.get("block_size", 512)
    )

    val_loader = create_sft_dataloader(
        jsonl_path=val_jsonl,
        batch_size=batch_size,
        shuffle=False,
        pad_id=TOKEN_PAD_ID,
        pad_target=TOKEN_IGNORE_TARGET,
        block_size=model_args.get("block_size", 512)
    )
    print(f"   ✅ DataLoader hazırlandı. Train batch sayısı: {len(train_loader)} | Val batch sayısı: {len(val_loader)}")

    # 5. Optimizer yapılandırması
    # SFT için optimizer durumunu sıfırdan başlatıyoruz (v1 base optimizer'ını taşımıyoruz)
    device_type = "cuda" if "cuda" in device else "cpu"
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    # 6. Başlangıç Değerlendirmesi
    print("\n🔍 Başlangıç Doğrulama Kaybı (Baseline SFT Öncesi) Hesaplanıyor...")
    val_loss_init = estimate_loss(model, val_loader, device=device)
    train_loss_init = estimate_loss(model, train_loader, device=device)
    print(f"   Başlangıç Train Loss : {train_loss_init:.4f}")
    print(f"   Başlangıç Val Loss   : {val_loss_init:.4f}")

    best_val_loss = val_loss_init
    train_iter = iter(train_loader)

    # Optimizer adımının ağırlıkları değiştirdiğini kanıtlamak için referans parametre
    ref_param = model.transformer.h[0].mlp.c_proj.weight
    ref_weights_before_opt = ref_param.detach().clone()
    optimizer_step_confirmed = False

    training_stats = {
        "initial_train_loss": train_loss_init,
        "initial_val_loss": val_loss_init,
        "final_train_loss": None,
        "final_val_loss": None,
        "best_val_loss": best_val_loss,
        "steps_completed": 0,
        "optimizer_updated": False,
        "saved_checkpoints": []
    }

    # 7. Eğitim Döngüsü
    print("\n" + "-" * 80)
    print(f"{'Adım':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'LR':<10} | {'Durum'}")
    print("-" * 80)

    t0 = time.time()
    last_train_loss = train_loss_init

    for step in range(1, max_iters + 1):
        # LR güncellemesi
        lr = get_lr(step - 1, max_iters, warmup_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            step_loss += loss.item()

            if not torch.isfinite(loss):
                raise FloatingPointError(f"❌ Adım {step}'de sonlu olmayan (NaN/Inf) loss tespit edildi!")

            loss.backward()

        # Gradyan kırpma (Gradient Clipping)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Parametrelerin güncellendiğini ölçmek için adım öncesi ağırlık kopyala
        if step == 1:
            param_before = ref_param.detach().clone()

        # Optimizer adımı
        optimizer.step()

        # 1. adımda optimizer kontrolü
        if step == 1:
            param_after = ref_param.detach().clone()
            delta = (param_after - param_before).abs().sum().item()
            if delta > 0:
                optimizer_step_confirmed = True
                print(f"      [Doğrulama] Optimizer ağırlıkları güncelledi! Δ = {delta:.8f}")

        last_train_loss = step_loss

        # Periyodik Evaluation ve Checkpoint
        status_msg = ""
        val_loss_curr = None
        if step % eval_interval == 0 or step == max_iters:
            val_loss_curr = estimate_loss(model, val_loader, device=device)
            if val_loss_curr < best_val_loss:
                best_val_loss = val_loss_curr
                ckpt_save_path = os.path.join(out_dir, "ckpt_best.pt")
                save_payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": step,
                    "best_val_loss": best_val_loss,
                    "config": {
                        "learning_rate": learning_rate,
                        "min_lr": min_lr,
                        "max_iters": max_iters,
                        "batch_size": batch_size,
                        "stage": "sft_pilot"
                    }
                }
                torch.save(save_payload, ckpt_save_path)
                training_stats["saved_checkpoints"].append(ckpt_save_path)
                status_msg = f"💾 Checkpoint Kaydedildi ({best_val_loss:.4f})"
            else:
                status_msg = "Val loss düşmedi (kayıt yok)"

        val_str = f"{val_loss_curr:.4f}" if val_loss_curr is not None else "-"
        print(f"{step:<6} | {step_loss:<12.4f} | {val_str:<12} | {lr:<10.2e} | {status_msg}")

    dt = time.time() - t0
    print("-" * 80)
    print(f"⏱️ Eğitim süresi: {dt:.2f} saniye ({dt/max_iters:.2f} s/adım)")

    training_stats["final_train_loss"] = last_train_loss
    training_stats["final_val_loss"] = best_val_loss
    training_stats["steps_completed"] = max_iters
    training_stats["optimizer_updated"] = optimizer_step_confirmed

    # 8. Kaydedilen Checkpoint'in Geri Yüklenme Testi
    if training_stats["saved_checkpoints"]:
        last_saved_ckpt = training_stats["saved_checkpoints"][-1]
        print(f"\n🧪 Kaydedilen Checkpoint Yükleme Testi: {last_saved_ckpt}")
        loaded = torch.load(last_saved_ckpt, map_location="cpu", weights_only=False)
        assert "model" in loaded and "best_val_loss" in loaded, "Kaydedilen checkpoint eksik anahtarlar içeriyor!"
        print(f"   ✅ Checkpoint sağlam! Kaydedilen iterasyon: {loaded['iter_num']}, Val Loss: {loaded['best_val_loss']:.4f}")

    # 9. Orijinal Checkpoint Dokunulmazlık Doğrulaması
    sha_after = calculate_sha256(init_ckpt)
    print(f"\n🔒 Orijinal Checkpoint SHA256 (Eğitim Sonrası): {sha_after}")
    assert sha_before == sha_after, "❌ ALARM: Orijinal v1 checkpoint değiştirildi!"
    print("   ✅ Orijinal v1 checkpoint KESİNLİKLE DEĞİŞTİRİLMEDİ (SHA256 birebir eşleşti).")

    return training_stats


def main():
    parser = argparse.ArgumentParser(description="Cores-AI TR v2.0 SFT Eğitim Motoru")
    parser.add_argument("--init_ckpt", type=str, default=os.path.join("out-cores-85m-test", "ckpt.pt"),
                        help="Başlangıç model ağırlıkları yolu")
    parser.add_argument("--out_dir", type=str, default="TESTLER/temp_sft_pilot",
                        help="Model kayıt çıktı dizini")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "sft_test"),
                        help="SFT veri dizini (train.jsonl, val.jsonl)")
    parser.add_argument("--max_iters", type=int, default=5,
                        help="Maksimum eğitim iterasyon sayısı")
    parser.add_argument("--eval_interval", type=int, default=2,
                        help="Doğrulama ve checkpoint değerlendirme sıklığı")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Başlangıç öğrenme oranı (varsayılan 2e-5)")
    parser.add_argument("--min_lr", type=float, default=2e-6,
                        help="Minimum öğrenme oranı")
    parser.add_argument("--warmup_iters", type=int, default=1,
                        help="Isınma iterasyonu sayısı")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch boyutu")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradyan biriktirme adımları")
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' veya 'cuda'")
    args = parser.parse_args()

    train_sft(
        init_ckpt=args.init_ckpt,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        device=args.device
    )


if __name__ == "__main__":
    main()
