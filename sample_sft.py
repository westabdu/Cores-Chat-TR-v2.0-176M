"""
Cores-AI TR v2.0 — SFT Terminal Sohbet Arayüzü (sample_sft.py)
==============================================================

Kullanım:
    # 1. İnteraktif Sohbet Modu:
    .venv/Scripts/python sample_sft.py --device cpu

    # 2. Tek Komut / Test Modu (Smoke Test):
    .venv/Scripts/python sample_sft.py --single_prompt "Merhaba!" --device cpu

Özellikler:
    - [BOS]Kullanıcı: ...\nAsistan: ...[EOS] formatını uygular.
    - EOS tokenı (ID: 3) üretildiğinde anında durur.
    - block_size (512) sınırını aşmamak için çok turlu konuşmalarda eski mesajları dinamik budar.
    - Sıcaklık (temperature), top_k ve maksimum token parametrelerini destekler.
"""

import os
import sys
import argparse
from typing import List, Tuple

import torch
from tokenizers import Tokenizer
from model import GPTConfig, GPT

# Windows UTF-8 konsol desteği
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_model_and_tokenizer(ckpt_dir_or_path: str, tokenizer_path: str, device: str):
    """Model ve Tokenizer'ı doğrulanmış şekilde yükler."""
    if os.path.isdir(ckpt_dir_or_path):
        ckpt_path = os.path.join(ckpt_dir_or_path, "ckpt.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir_or_path, "ckpt_best.pt")
    else:
        ckpt_path = ckpt_dir_or_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint dosyası bulunamadı: {ckpt_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer dosyası bulunamadı: {tokenizer_path}")

    print(f"📦 Model yükleniyor: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]

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

    print(f"📖 Tokenizer yükleniyor: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    assert bos_id is not None, "[BOS] tokenı bulunamadı!"
    assert eos_id is not None, "[EOS] tokenı bulunamadı!"

    param_count = model.get_num_params()
    print(f"✅ Model hazır ({param_count:,} parametre). Vocab: {tokenizer.get_vocab_size()}")

    return model, tokenizer, bos_id, eos_id, model_args.get("block_size", 512)


def generate_response(
    model: GPT,
    tokenizer: Tokenizer,
    conversation_history: List[Tuple[str, str]],
    new_user_message: str,
    bos_id: int,
    eos_id: int,
    block_size: int = 512,
    device: str = "cpu",
    temperature: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 128
) -> Tuple[str, bool]:
    """
    Çok turlu konuşma geçmişini [BOS]Kullanıcı: ...\nAsistan: ...[EOS] formatında derler.
    Bağlam penceresi 512 token sınırını aşarsa eski turları otomatik kırpar.
    """
    history_tokens = []
    
    # Geçmiş turları ekle
    for u_msg, a_msg in conversation_history:
        turn_text = f"Kullanıcı: {u_msg}\nAsistan: {a_msg}[EOS]\n"
        history_tokens.extend(tokenizer.encode(turn_text).ids)

    # Yeni kullanıcı mesajı
    current_turn_prefix = f"Kullanıcı: {new_user_message}\nAsistan:"
    current_tokens = tokenizer.encode(current_turn_prefix).ids

    # 512 bağlam sınırına göre bütçeleme: [BOS] (1 token) + context + max_new_tokens <= block_size
    allowed_context = block_size - max_new_tokens - 2
    if len(current_tokens) > allowed_context:
        # Tek mesaj çok uzunsa kırp
        current_tokens = current_tokens[-allowed_context:]
        total_prompt_ids = [bos_id] + current_tokens
    else:
        # Geçmiş mesajları sığdığı kadar geriden ekle
        remaining_budget = allowed_context - len(current_tokens)
        trimmed_history = history_tokens[-remaining_budget:] if remaining_budget > 0 else []
        total_prompt_ids = [bos_id] + trimmed_history + current_tokens

    x = torch.tensor(total_prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = len(total_prompt_ids)

    with torch.no_grad():
        device_type = "cuda" if "cuda" in device else "cpu"
        ptdtype = torch.float16 if device_type == "cuda" else torch.float32
        ctx = (torch.amp.autocast(device_type=device_type, dtype=ptdtype)
               if device_type == "cuda" else torch.no_grad())
        with ctx:
            y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    generated_all = y[0].tolist()
    new_ids = generated_all[prompt_len:]

    eos_found = False
    if eos_id in new_ids:
        eos_pos = new_ids.index(eos_id)
        cut_ids = new_ids[:eos_pos]
        eos_found = True
    else:
        cut_ids = new_ids

    response_text = tokenizer.decode(cut_ids).strip()
    return response_text, eos_found


def main():
    parser = argparse.ArgumentParser(description="Cores-AI TR v2.0 SFT Terminal Sohbet")
    parser.add_argument("--out_dir", type=str, default="out-cores-85m-sft-v2",
                        help="Model checkpoint dizini")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_tr_v2.json",
                        help="Tokenizer dosyası")
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cpu' veya 'cuda'")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Örnekleme sıcaklığı (varsayılan: 0.7)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K örnekleme (varsayılan: 50)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Üretilecek azami token sayısı (varsayılan: 128)")
    parser.add_argument("--single_prompt", type=str, default=None,
                        help="Tek seferlik test istemi (smoke test için)")
    args = parser.parse_args()

    model, tokenizer, bos_id, eos_id, block_size = load_model_and_tokenizer(
        ckpt_dir_or_path=args.out_dir,
        tokenizer_path=args.tokenizer,
        device=args.device
    )

    # 1. Tek Seferlik Test Modu (Smoke Test)
    if args.single_prompt:
        print(f"\n[Smoke Test] Kullanıcı: {args.single_prompt}")
        resp, eos_hit = generate_response(
            model=model,
            tokenizer=tokenizer,
            conversation_history=[],
            new_user_message=args.single_prompt,
            bos_id=bos_id,
            eos_id=eos_id,
            block_size=block_size,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens
        )
        print(f"Asistan: {resp}")
        print(f"(Durum: {'EOS ile sonlandı' if eos_hit else 'Maksimum token sınırında kesildi'})\n")
        return

    # 2. İnteraktif Sohbet Modu
    print("\n" + "=" * 70)
    print("🇹🇷 Cores-AI TR v2.0 (SFT) — İnteraktif Sohbet Konsolu")
    print("=" * 70)
    print("Komutlar:")
    print("  - Çıkış için: 'q', 'exit' veya 'quit'")
    print("  - Konuşma geçmişini sıfırlamak için: 'clear'")
    print("=" * 70 + "\n")

    conversation_history: List[Tuple[str, str]] = []

    while True:
        try:
            user_input = input("\n👤 Kullanıcı: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["q", "exit", "quit"]:
                print("👋 Görüşmek üzere!")
                break

            if user_input.lower() == "clear":
                conversation_history.clear()
                print("🧹 Konuşma geçmişi temizlendi.")
                continue

            print("🤖 Asistan düşünülüyor...", end="\r", flush=True)

            response, eos_hit = generate_response(
                model=model,
                tokenizer=tokenizer,
                conversation_history=conversation_history,
                new_user_message=user_input,
                bos_id=bos_id,
                eos_id=eos_id,
                block_size=block_size,
                device=args.device,
                temperature=args.temperature,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens
            )

            # Ekrandaki "düşünülüyor" ifadesini temizle
            print(" " * 30, end="\r", flush=True)
            print(f"🤖 Asistan: {response}")

            # Konuşma geçmişine ekle
            conversation_history.append((user_input, response))

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Görüşmek üzere!")
            break


if __name__ == "__main__":
    main()
