# Cores-AI TR 176M — Model Yapılandırması (Aşama 1)
# ==============================================================================
# Bu yapılandırma dosyası, nanoGPT/configurator.py akışıyla uyumlu olarak
# train.py içindeki global değişkenleri 176M mimarisine göre günceller.
#
# Hedef Mimari:
#   - Parametre: ~176.021.248 (toplam benzersiz)
#   - Katman (n_layer): 18
#   - Kafa (n_head): 13 (Kafa başına boyut: 832 / 13 = 64)
#   - Embedding (n_embd): 832
#   - Bağlam (block_size): 512
#   - Sözlük (vocab_size): 31104 (GPU verimliliği için 64'ün katı: 486 * 64)
#   - Bias: True
#   - Gradient Checkpointing: True (GTX 1650 4 GB VRAM tasarrufu)
#   - Ağırlık Paylaşımı: Aktif (wte <-> lm_head)
# ==============================================================================

# I/O ve Günlükleme
out_dir = 'out-cores-176m'
eval_interval = 1000
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Wandb (varsayılan kapalı)
wandb_log = False
wandb_project = 'cores-ai-tr'
wandb_run_name = 'cores-176m'

# Veri ve Batch Boyutları (GTX 1650 4 GB VRAM odaklı)
dataset = 'turkce'
batch_size = 1                   # Mikro-batch boyutu: 1 (OOM önleme)
gradient_accumulation_steps = 64 # Etkin batch: 1 * 64 = 64 dizi (32.768 token/iter)
block_size = 512                 # Bağlam uzunluğu

# 176M Model Mimarisi
n_layer = 18
n_head = 13
n_embd = 832
vocab_size = 31104
bias = True
dropout = 0.0
use_gradient_checkpointing = True

# AdamW Optimizer
learning_rate = 3e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Öğrenme Oranı Çizelgesi (Cosine Decay with Warmup)
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 100000
min_lr = 3e-5

# Sistem, Donanım ve Hassasiyet
device = 'cuda'
# GTX 1650 donanımsal bfloat16 desteklemez; float16 GradScaler ile kullanılır
dtype = 'float16'
# Windows ortamında MSVC/Triton bağımlılık riskini önlemek için compile kapalı
compile = False
