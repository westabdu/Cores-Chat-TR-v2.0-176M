"""
Cores-AI TR - Türkçe Metin Üretme (Inference) Scripti
"""

import os
import argparse
import torch
from tokenizers import Tokenizer
from model import GPTConfig, GPT

DEFAULT_OUT_DIR = 'out-cores-152m'
DEFAULT_TOKENIZER = 'tokenizer_tr_v2.json'
DEFAULT_START = 'Türkiye'
DEFAULT_NUM_SAMPLES = 5
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 200
DEFAULT_DEVICE = 'cuda'
DEFAULT_DTYPE = 'float16'
DEFAULT_SEED = 1337


def main():
    parser = argparse.ArgumentParser(description='Cores-AI TR - Türkçe Metin Üretme')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument('--start', type=str, default=DEFAULT_START)
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--dtype', type=str, default=DEFAULT_DTYPE)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.no_grad()

    print(f"📦 Tokenizer yükleniyor: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    bos_id = tokenizer.token_to_id("[BOS]")
    print(f"   Vocab: {vocab_size}, [BOS] ID: {bos_id}")

    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    print(f"\n🧠 Model yükleniyor: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint bulunamadı: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model_args = checkpoint['model_args']
    if 'use_gradient_checkpointing' in model_args:
        model_args['use_gradient_checkpointing'] = False

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    iter_num = checkpoint.get('iter_num', '?')
    best_val_loss = checkpoint.get('best_val_loss', '?')
    print(f"   ✅ Model yüklendi! iter={iter_num}, best_val_loss={best_val_loss}")
    print(f"   Parametre: {model.get_num_params():,}")

    print(f"\n🔤 Prompt: \"{args.start}\"")
    encoded = tokenizer.encode(args.start)
    start_ids = encoded.ids
    if bos_id is not None:
        start_ids = [bos_id] + start_ids

    x = torch.tensor(start_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    print(f"\n{'=' * 60}")
    print(f"🚀 {args.num_samples} örnek üretiliyor...")
    print(f"{'=' * 60}")

    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
                generated_ids = y[0].tolist()
                text = tokenizer.decode(generated_ids)
                print(f"\n--- Örnek {k+1}/{args.num_samples} ---")
                print(text)
                print()

    print(f"{'=' * 60}")
    print(f"✅ Üretim tamamlandı!")


if __name__ == "__main__":
    main()
