# Cores-AI TR 176M — Google Colab GPU Yapılandırması
# ==============================================================================
# Bu yapılandırma dosyası, Google Colab ortamındaki NVIDIA T4 (16 GB VRAM)
# donanımı için özel olarak ayarlanmıştır.
#
# Avantajlar:
#   - Checkpoint'ler doğrudan Google Drive'a kalıcı olarak kaydedilir.
#   - 16 GB VRAM sayesinde mikro-batch boyutu 4'e çıkarılmıştır.
#   - Gradient checkpointing ile VRAM tüketimi minimumda tutulur.
#   - T4 Tensor Core donanımında float16 AMP ile yüksek hız sağlanır.
# ==============================================================================

# I/O ve Kalıcı Depolama (Google Drive)
out_dir = '/content/drive/MyDrive/Cores-176M/out-cores-176m'
eval_interval = 250
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # Checkpoint'ten devam etmek için 'resume'

# Wandb Günlükleme (opsiyonel)
wandb_log = False
wandb_project = 'cores-ai-tr'
wandb_run_name = 'cores-176m-colab'

# Veri ve Batch Boyutları (Colab T4 16 GB VRAM için optimize)
dataset = 'cores176m'
batch_size = 4                   # Mikro-batch: 4 (T4 için güvenli ve hızlı)
gradient_accumulation_steps = 16 # Etkin batch: 4 * 16 = 64 dizi (32.768 token/iter)
block_size = 512                 # Bağlam uzunluğu (176M hedefi)

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

# Donanım ve Hassasiyet (Colab T4)
device = 'cuda'
dtype = 'float16'  # NVIDIA T4 Tensor Cores için float16 AMP
compile = False    # Smoke test ve başlangıç için False (Linux'ta istenirse True açılabilir)
