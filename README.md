# 🇹🇷 Cores-AI TR — v1.0.0 (Base) & v2.0.0 (SFT Asistan)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Model Parameters](https://img.shields.io/badge/Parametre-81.17M-green.svg)](#-model-mimarisi)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Model_Ağırlıkları-FFD21E.svg)](https://huggingface.co/aapo33/cores-ai-tr-v1.0.0-85m)
[![Vocab Size](https://img.shields.io/badge/Vocab-50%2C000-orange.svg)](#-tokenizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: SFT v2.0 Ready](https://img.shields.io/badge/SFT_v2.0-Tamamlandı-success.svg)](#-cores-ai-tr-v20-sft-dönüşümü)
[![Durum Raporu](https://img.shields.io/badge/📑_PDF_Durum_Raporu-İndir-red.svg)](Cores_AI_TR_v2_Durum_Raporu.pdf)

**Cores-AI TR**, tamamen Türkçe korpuslar üzerinde **sıfırdan (from scratch)** eğitilmiş, açık kaynaklı ve hafif siklet (81.17M parametre) bir Türkçe Üretken Dil Modelidir (GPT).

Bu depo; ham metin tamamlama yeteneğine sahip temel modelin (**v1 Base Model**) sıfırdan ön eğitimini, ardından diyalog ve konuşma kabiliyeti kazandırılan **v2 SFT (Supervised Fine-Tuning)** sürümünü, kapsamlı benchmark testlerini ve interaktif terminal sohbet arayüzünü (`sample_sft.py`) içerir.

---

## 🔥 Büyük Güncelleme: v2.0.0 SFT Devrimi (Ham Metinden Türkçe Asistana!)

> **Normalde yalnızca sıfırdan ön eğitilmiş bir ham metin tamamlama modeli (v1 Base) olarak kalması planlanan Cores-AI TR 85M**, açık kaynak Türkçe yapay zeka topluluğuna gerçek bir katkı sunmak amacıyla radikal bir kararla **Supervised Fine-Tuning (SFT)** sürecine alındı.
> 
> Yapılan bu büyük güncelleme ile model, ansiklopedi maddeleri tamamlayan bir araştırma prototipinden; **kullanıcıyla canlı diyalog kurabilen, selamlamalara kibarlıkla karşılık veren, şablon kirliliğinden arınmış ve işi bitince susmayı bilen gerçek bir Türkçe Asistan Modeline** dönüştürüldü!

### 🌟 v2.0.0 İle Gelen Başlıca Yenilikler

* 🗣️ **Asistan Kimliği ve Diyalog Yeteneği:** Model artık rastgele film listesi basmak yerine doğrudan kullanıcıya hitap eden `[BOS]Kullanıcı: ...\nAsistan: ...[EOS]` protokolünü benimsedi.
* 🧹 **Vikipedi ve Çöp Metin Kirliliği Sıfırlandı (%54.7 ➔ %0.0):** v1 sürümünde sıkça karşılaşılan wikitext şablonları (`style="background:#F1F1F1"`, `colspan`, `Kaynakça`) modelin çıktılarından **%100 oranında temizlendi.**
* 🛑 **Kusursuz Susma Disiplini (%93.3 EOS Oranı):** v1 modelinde soruların %43.3'ünde duramayıp 128 token dolana kadar anlamsızca devam etme takıntısı son buldu; model yanıtını tamamladığı anda `[EOS]` basarak durmayı öğrendi.
* 📉 **Validation Loss 4.62'den 3.99'a Geriledi:** Toplam 750 adımlık aşamalı eğitim (250 pilot + 500 devam) ile validation loss doyum noktasına ulaştı (3.9976) ve hiçbir aşırı öğrenme (overfitting) yaşanmadı.
* 💬 **İnteraktif Terminal Sohbet Konsolu (`sample_sft.py`):** Modeli tek komutla bilgisayarınızda (NVIDIA GPU veya sadece CPU ile!) canlı olarak çalıştırabileceğiniz çok turlu interaktif bir sohbet arayüzü eklendi.
* 🔬 **150 Soruluk Şeffaf Benchmark & Raporlama Paketi:** 6 temel kategoride 3 model neslinin (v1, Pilot, Final) kıyaslandığı bağımsız test süiti, karşılaştırma araçları (`tools/`) ve tek tıkla derlenen grafikli PDF durum raporu depoya kazandırıldı.
* 📦 **Ayrık Dağıtım Mimarisi (GitHub + Hugging Face Hub):** Kod tabanı GitHub'da, 978 MB'lık model ağırlıkları ise Hugging Face Hub'da (`aapo33/cores-ai-tr-v1.0.0-85m`) açık paylaşıma sunuldu.

---

> [!IMPORTANT]
> ### 📢 GERÇEKÇİ MODEL DEĞERLENDİRMESİ VE BİLİNEN SINIRLAMALAR
> **Cores-AI TR 85M**, modern açık kaynaklı büyük dil modelleri (7B, 13B, 70B vb.) ile kıyaslanamaz; **81.17 milyon parametreli ultra hafif siklet bir araştırma modelidir.**
> * **Genel Bilgi ve Olgusal Sorular:** Model genel kültür, tarih, coğrafya veya fen sorularında **güvenilir bir bilgi kaynağı DEĞİLDİR.** Olgusal doğruluk gerektiren sorularda halüsinasyon (yanlış bilgi uydurma) riski yüksektir.
> * **Yankılama (Echoing):** *"Bana 3 meyve say"* veya *"Bana tek kelime söyle"* gibi kural ve liste içeren soyut talimatlarda, model soruyu çözmek yerine sorunun kendisini kullanıcıya geri yankılayabilir.
> * **Tekrarlama:** Agresif kısıtlama promptlarında nadiren de olsa kelime tekrarlarına girebilir.
> * **Asıl Başarısı:** SFT süreci sonucunda modelin **Vikipedi/tablo kirliliği %54.7'den %0.0'a sıfırlanmış**, model işi bitince susmayı (**%93.3 EOS durma oranı**) ve kullanıcıyla nazikçe Türkçe diyalog kurmayı öğrenmiştir.

---

## 📑 İçindekiler
1. [Büyük Güncelleme: v2.0.0 SFT Devrimi](#-büyük-güncelleme-v200-sft-devrimi-ham-metinden-türkçe-asistana)
2. [Hızlı Başlangıç & İnteraktif Sohbet](#-hızlı-başlangıç--interaktif-sohbet)
3. [Model Mimarisi](#-model-mimarisi)
4. [Model Checkpoint Sürümleri](#-model-checkpoint-sürümleri)
5. [Cores-AI TR v2.0 SFT Dönüşümü](#-cores-ai-tr-v20-sft-dönüşümü)
6. [Karşılaştırmalı Benchmark Değerlendirmesi](#-karşılaştırmalı-benchmark-değerlendirmesi)
7. [Temsilî Soru-Cevap Kıyaslamaları](#-temsilî-soru-cevap-kıyaslamaları)
8. [Benchmark Test Seti Detayı](#-benchmark-test-seti-detayı)
9. [Depo ve Klasör Yapısı](#-depo-ve-klasör-yapısı)
10. [Analiz ve Denetim Araçları](#-analiz-ve-denetim-araçları-tools)
11. [Kendi Modelinizi Eğitme & Geliştirme](#-kendi-modelinizi-eğitme--geliştirme)
12. [GitHub & Hugging Face Dağıtım Rehberi](#-github--hugging-face-dağıtım-rehberi)
13. [Lisans](#-lisans)

---

## 🚀 Hızlı Başlangıç & İnteraktif Sohbet

### 1. Kurulum
Gerekli Python paketlerini sanal ortamınıza yükleyin (Python 3.10+):

```bash
git clone https://github.com/KULLANICI_ADINIZ/Cores-AI-TR-85M.git
cd Cores-AI-TR-85M
pip install -r requirements.txt
```

### 2. Model Ağırlıklarını İndirme (Hugging Face'den)
Model checkpoint dosyaları (~978 MB) GitHub'a yüklenmez. Hugging Face'den indirmeniz gerekmektedir:

```bash
# Hugging Face Hub CLI ile otomatik indirme
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
# v2 SFT Model (Asistan)
hf_hub_download(repo_id='aapo33/cores-ai-tr-v1.0.0-85m', filename='out-cores-85m-sft-v2/ckpt.pt', local_dir='.')
hf_hub_download(repo_id='aapo33/cores-ai-tr-v1.0.0-85m', filename='out-cores-85m-sft-v2/ckpt_best.pt', local_dir='.')
# v1 Base Model (Opsiyonel)
hf_hub_download(repo_id='aapo33/cores-ai-tr-v1.0.0-85m', filename='out-cores-85m-test/ckpt.pt', local_dir='.')
"
```

### 3. İnteraktif Terminal Sohbetini Başlatın (v2 SFT)
Aşağıdaki komut, `out-cores-85m-sft-v2/` klasöründeki resmi v2 modelini ve özel Türkçe BPE tokenizer'ını yükleyerek çok turlu interaktif bir sohbet konsolu açar:

```bash
python sample_sft.py --device cpu
```

*(Eğer NVIDIA GPU'nuz varsa `--device cuda` parametresini kullanabilirsiniz.)*

**Konsol Komutları:**
* `q`, `exit` veya `quit`: Sohbetten çıkar.
* `clear`: Konuşma geçmişini sıfırlar.

### 4. Tek Seferlik Hızlı Test (Smoke Test)
Terminalden çıkmadan tek bir istemi hızlıca test etmek için:

```bash
python sample_sft.py --single_prompt "Merhaba!" --device cpu
```

---

## 🧠 Model Mimarisi

Cores-AI TR, Andrej Karpathy'nin NanoGPT tabanını temel alan saf bir **Decoder-Only Transformer** mimarisidir:

| Bileşen | Değer | Teknik Açıklama |
|---|---|---|
| **Toplam Parametre** | **81,165,440 (~81.17M)** | Tüm embedding ve dikkat katmanları dahil |
| **Katman Sayısı (`n_layer`)** | **10** | Derinlik |
| **Dikkat Kafası (`n_head`)** | **10** | Multi-Head Attention |
| **Gömme Boyutu (`n_embd`)** | **640** | Hidden representation size (Kafa başına 64 boyut) |
| **Bağlam Uzunluğu (`block_size`)** | **512 token** | Maksimum ardışık token dizilimi |
| **Kelime Dağarcığı (`vocab_size`)** | **50,000** | Byte-Pair Encoding (BPE) Türkçe sözlük |
| **Önyargı Terimi (`bias`)** | `False` | Daha hızlı ve bellek dostu çıkarım |
| **Aktivasyon Fonksiyonu** | `GELU` | Gaussian Error Linear Unit |
| **Dropout** | `0.0` | SFT ve çıkarım sırasında kapalı |

### 📦 Tokenizer
- **Dosya**: `tokenizer_tr_v2.json`
- **Özel Tokenlar**:
  - `[UNK]` = `0` (Bilinmeyen kelime)
  - `[PAD]` = `1` (Dolgu tokenı)
  - `[BOS]` = `2` (Dizi başı / Beginning of Sequence)
  - `[EOS]` = `3` (Dizi sonu / End of Sequence)
- **Diyalog Şablonu**:
  ```text
  [BOS]Kullanıcı: {soru}\nAsistan: {yanıt}[EOS]
  ```

---

## 💾 Model Checkpoint Sürümleri

Depoda iki farklı model ağırlığı bulunmaktadır:

| Sürüm | Checkpoint Yolu | İterasyon | En İyi Val Loss | SHA-256 Özeti | Durum |
|---|---|:---:|:---:|---|---|
| **v1.0 Base** | `out-cores-85m-test/ckpt.pt` | 5.000 | 2.8424 | `e88f284f...47e51` | 🔒 Salt-Okunur Ham Ön Eğitim |
| **v2.0 SFT** | `out-cores-85m-sft-v2/ckpt.pt` | 750 (Küm.) | **3.9976** | `3fc057b0...71618` | 🟢 **Resmi v2 Asistan Sürümü** |

**SHA-256 Tam Değerler:**
- **v1 Base:** `e88f284fdfb9ddc48075a97e47f509859319590df312a4226d9b355c63347e51`
- **v2 SFT:** `3fc057b04e3b949291b5dc249ba420f511d868cf88b2a4da29c884315cb71618`
- **Pilot (250 Adım):** `106343294c6af89f3bc9201b6a51a78c527b2657cd3cdfc4d2942c03e0d8bb6a`

*Not: Büyük ağırlık dosyaları (~978 MB) GitHub kota sınırları nedeniyle repoda tutulmamakta, [Hugging Face](https://huggingface.co/aapo33/cores-ai-tr-v1.0.0-85m) üzerinden dağıtılmaktadır.*

---

## 🔄 Cores-AI TR v2.0 SFT Dönüşümü

### 1. Neden SFT Yapıldı?
v1 temel modeli Türkçe söz dizimi ve kelime haznesini bilmesine rağmen saf bir metin tamamlayıcısıydı. Kullanıcı "Merhaba!" yazdığında bir diyalog kurmak yerine Vikipedi sayfaları, film oyuncu kadroları veya HTML tablo kodları (`style="background:#F1F1F1"`, `colspan`, `Kaynakça`) döküyor ve ne zaman susacağını kestiremiyordu (%43.3 duramama oranı).

SFT (Supervised Fine-Tuning) sürecinin temel amacı:
1. Ansiklopedi metin tamamlayıcısı kimliğini kırıp **"Türkçe Asistan"** rolü kazandırmak.
2. İşini bitirdiği anda `[EOS]` (End-of-Sequence) tokenı üreterek **kibarca susmasını sağlamak**.
3. Çıktılardaki Vikipedi, viki-tablo ve kaynakça gibi format kirliliklerini **sıfırlamak**.

---

### 2. SFT Veri Kümesi Dağılımı (8.936 Temiz Örnek)

Üç farklı açık kaynaklı Türkçe veri kümesinden kalite filtreleri, 512 token uzunluk denetimi ve **150 soruluk benchmark sızıntı izolasyonu** uygulanarak **8.936 temiz diyalog çifti** derlenmiştir:

<p align="center">
  <img src="TESTLER/charts/dataset_composition.png" alt="SFT Veri Kümesi Kaynak Dağılımı" width="750">
  <br>
  <em><b>Şekil 1:</b> 8.936 Örnekten Oluşan Temiz SFT Veri Kümesi Kaynak Dağılımı ve Train/Val Ayrımı (%90 / %10)</em>
</p>

| Kaynak Veri Kümesi | Lisans | Taranan | Kabul Edilen | Boş/Bozuk | Yinelenen | Benchmark Sızıntısı | 512+ Token Aşımı |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `sixfingerdev/turkish-qa-multi-dialog-dataset` | MIT | 5.081 | **5.000** | 26 | 51 | 3 (İzole edildi) | 1 |
| `ogulcanaydogan/Turkish-LLM-v10-Training` | Apache-2.0 | 4.920 | **3.500** | 0 | 114 | 0 | 1.306 |
| `AlicanKiraz0/Turkish-SFT-Dataset-v1.0` | MIT | 5.579 | **436** | 2 | 0 | 0 | 5.141 |
| **GENEL TOPLAM** | | **15.580** | **8.936** | **28** | **165** | **3 (Ayıklandı)** | **6.448** |

* **Bölümleme:** %90 Eğitim Kümesi (8.043 örnek), %10 Doğrulama Kümesi (893 örnek).
* **Güvenlik Güvencesi:** Eğitim ve doğrulama kümeleri arasında tek bir ortak örnek bulunmadığı hash kontrolüyle doğrulanmıştır.

---

### 3. Loss Masking Mekanizması

Modelin kullanıcının sorduğu soruyu ezberleyerek kayıp fonksiyonunu (loss) aldatmasını önlemek için PyTorch `CrossEntropyLoss(ignore_index=-1)` maskelemesi uygulanmıştır:

```text
Girdi Dizilimi:  [BOS] Kullanıcı: Türkiye'nin başkenti neresidir? \n Asistan: Ankara'dır. [EOS]
Hedef Maskesi:     -1      -1       -1   -1   -1   -1   -1   -1   -1      -1    [Ankara] ['dır] [.] [EOS]
                                                                                ▲ YALNIZCA BU KISIM EĞİTİLİR
```
Kullanıcı istemi ve şablon etiketleri `-1` yapılarak gradyan hesaplamasının dışında tutulmuş; model **yalnızca asistanın cevabı ve sonundaki `[EOS]` tokenı için ağırlık güncellemiştir**.

---

### 4. Çok Aşamalı Eğitim Dinamikleri (750 Kümülatif Adım)

Eğitim iki kontrollü fazda yürütülmüştür:
1. **Faz 1 (Pilot - 250 Adım):** Orijinal v1 checkpoint'i dondurulmadan temel diyalog formatı ve susma refleksi kazandırıldı (`lr = 2e-5 ➔ 2e-6`).
2. **Faz 2 (Aşama 7 - 500 Adım):** Pilot checkpoint'ten devam edilerek daha düşük öğrenme oranıyla (`lr = 1e-5 ➔ 1e-6`) ağırlık bozulması (catastrophic forgetting) engellendi ve validation loss **3.9976** seviyesine indirildi.

<p align="center">
  <img src="TESTLER/charts/val_loss_curve_complete.png" alt="750 Adımlık Tam Eğitim Kaybı Eğrisi" width="750">
  <br>
  <em><b>Şekil 2:</b> Faz 1 Pilot (250 Adım) ve Faz 2 Devam Eğitimi (500 Adım) Boyunca Kesintisiz Validation Loss Eğrisi (4.6248 ➔ 3.9976)</em>
</p>

| Aşama | Adım Aralığı | Başlangıç Val Loss | Bitiş Val Loss | LR Aralığı | Efektif Batch | Süre & Cihaz |
|---|---|:---:|:---:|---|:---:|:---:|
| **Faz 1: SFT Pilot** | 0 ➔ 250 | 4.6248 | 4.0849 | 2e-5 ➔ 2e-6 (cosine) | 4 × 4 = 16 | ~40 dk (CPU) |
| **Faz 2: Devam Eğitimi** | 0 ➔ 500 (250-750) | 4.0849 | **3.9976** | 1e-5 ➔ 1e-6 (cosine) | 4 × 4 = 16 | ~75 dk (CPU) |
| **KÜMÜLATİF TOPLAM** | **750 Adım** | **4.6248** | **3.9976** | — | **16** | **~115 dk (CPU)** |

> [!NOTE]
> Grafikte görüldüğü üzere 750 adım boyunca hiçbir aşamada **aşırı öğrenme (overfitting) emaresi görülmemiştir**. Validation loss her 50 adımda bir düzenli ve kararlı biçimde gerilemiştir.

---

## 📊 Karşılaştırmalı Benchmark Değerlendirmesi

Sabit **150 soruluk benchmark** (`TESTLER/benchmark_questions.json`) üzerinde 3 model nesli özdeş test koşullarında (`temperature=0.7`, `top_k=50`, `max_tokens=128`, `seed=42`, `device=cpu`) karşılaştırılmıştır:

### 1. Genel Metrik Karşılaştırması

<p align="center">
  <img src="TESTLER/charts/error_reduction_3way.png" alt="Üç Nesil Hata ve Kirlilik Azalması" width="750">
  <br>
  <em><b>Şekil 3:</b> Üç Model Neslinde Patolojik Hataların, Vikipedi Kirliliğinin ve Tekrar Döngülerinin Çarpıcı Düşüşü</em>
</p>

| Değerlendirme Ölçütü | v1 Base (Ham Model) | v2 Pilot (250 Adım) | v2 Final (750 Adım) | Başarı ve Değişim |
|---|:---:|:---:|:---:|---|
| **EOS Vurma (Doğru Durma)** | %56.7 (85/150) | **%94.7** (142/150) | **%93.3** (140/150) | 🟢 **+%36.6 artış** (Susmayı öğrendi) |
| **Vikipedi / Tablo / HTML Kirliliği** | %54.7 (82 soru) | %1.3 (2 soru) | **%0.0 (0 soru)** | 🟢 **%100 TAMAMEN SIFIRLANDI!** |
| **Ağır Tekrar Döngüsü** | %26.7 (40 soru) | %10.0 (15 soru) | **%9.3 (14 soru)** | 🟢 **%65.2 Azalma** |
| **Maksimum Token Kesilmesi (Cut)** | %43.3 (65 soru) | %5.3 (8 soru) | **%6.7 (10 soru)** | 🟢 Kontrollü ve özlü bitiş |
| **Ortalama Üretilen Token Sayısı** | ~85.0 token (susamıyor) | 25.9 token | **24.7 token** | 🟢 Odaklı ve net cevaplar |

---

### 2. Kategori Bazında EOS Oranları Karşılaştırması

<p align="center">
  <img src="TESTLER/charts/eos_comparison_3way.png" alt="Kategori Bazında 3 Nesil EOS Oranları" width="750">
  <br>
  <em><b>Şekil 4:</b> 6 Temel Benchmark Kategorisi Bazında 3 Model Neslinin [EOS] ile Düzgün Durma Başarı Oranları</em>
</p>

| Kod | Kategori Adı | Soru | v1 Base EOS | v2 Pilot EOS | v2 Final EOS (750 Adım) | Net Değişim | Niteliksel Gözlem |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **A** | **Selamlaşma ve Rol Tanıma** | 25 | %48.0 (12) | %100.0 (25) | **%88.0 (22)** | 🟢 **+40.0%** | Doğrudan selam verme ve rolü benimseme |
| **B** | **Türkçe Cümle Tamamlama** | 30 | %53.3 (16) | %100.0 (30) | **%93.3 (28)** | 🟢 **+40.0%** | Akıcı, gramere ve ek uyumuna sadık tamamlama |
| **C** | **Basit Soru-Cevap (Faktüel)**| 25 | %72.0 (18) | %96.0 (24) | **%96.0 (24)** | 🟢 **+24.0%** | Yanıt formatı oturdu, halüsinasyon sürüyor |
| **D** | **Talimat Takibi** | 25 | %44.0 (11) | %84.0 (21) | **%92.0 (23)** | 🟢 **+48.0%** | Pilota göre **+%8.0 daha iyi durma** |
| **E** | **Kısa Özetleme** | 25 | %72.0 (18) | %96.0 (24) | **%100.0 (25)** | 🟢 **+28.0%** | **Sıfır takılma, %100 kusursuz bitiş** |
| **F** | **Tekrar ve Tutarlılık** | 20 | %50.0 (10) | %90.0 (18) | **%90.0 (18)** | 🟢 **+40.0%** | Kararlı durma disiplini korundu |
| **TOPLAM** | **Genel Benchmark Ortalaması** | **150** | **%56.7 (85)** | **%94.7 (142)** | **%93.3 (140)** | 🟢 **+36.6%** | **Belirgin ve kalıcı asistan dönüşümü** |

---

## 🔍 Temsilî Soru-Cevap Kıyaslamaları

Aşağıdaki karşılaştırma tablosu, modelin kontrolsüz bir metin tamamlayıcısından odaklı bir asistana evrilişini satır satır göstermektedir:

| ID & Soru | v1 Base (Ham Model) | v2 Pilot (250 Adım) | v2 Final (750 Adım) | Gelişim Analizi |
|---|---|---|---|---|
| **A01**<br>*"Merhaba!"* | `Merhaba! (The Old Men) (Türkçe: Merhaba) Türkçe: Sensizler...`<br>*(❌ 128 tok kesildi)* | `Merhaba!`<br>*(✅ EOS — 3 token)* | `Merhaba!`<br>*(✅ EOS — 3 token)* | Ansiklopedi film maddesi dökme takıntısı kırıldı; temiz ve kibar selamlama oturdu. |
| **A02**<br>*"Sen kimsin?"* | `Sen kimsin? (dizi) Sen Kimsin?, 2012 yapımı Türk sinema filmi...`<br>*(❌ 128 tok kesildi)* | `Sen ne kadar mutlu olmuyor ki, tam bir umutta...`<br>*(✅ EOS — 31 token)* | `Sen kimsin?`<br>*(✅ EOS — 4 token)* | Film maddesi basmayı bıraktı; ancak soyut kimlik sorularında kullanıcıyı yankılama eğilimi var. |
| **B02**<br>*"Türkiye'nin en büyük şehri"* | `Türkiye'nin en büyük şehri olan İstanbul'un nüfusu 15 milyondur. Kaynakça...`<br>*(❌ 128 tok kesildi)* | `İstanbul'un en büyük şehri`<br>*(✅ EOS — 6 token)* | `İstanbul'un en büyük şehri`<br>*(✅ EOS — 6 token)* | Doğru şehri (İstanbul) anında tespit ediyor ve kaynakça çöpleri olmadan net bitiriyor. |
| **C01**<br>*"Türkiye'nin başkenti neresidir?"* | `Türkiye'nin başkenti neresidir? Türkiye Cumhuriyeti Cumhurbaşkanlığı...`<br>*(❌ 128 tok kesildi)* | `Kıbrıs'ın başkenti neresidir`<br>*(✅ EOS — 6 token)* | `Türkiye'nin başkenti neresidir`<br>*(✅ EOS — 6 token)* | Pilot sürümdeki yanlış ülke ("Kıbrıs") sapması 750 adımda doğru ülkeye ("Türkiye") odaklandı. |
| **C02**<br>*"Gökyüzü hangi renktir?"* | `Gökyüzü hangi renktir? Gökyüzü Mavisi (film) Gökyüzü...`<br>*(❌ 128 tok kesildi)* | `Gökyüzü, Dünya'nın en iyi beyaz renktir.`<br>*(✅ EOS — 12 token)* | `Gökyüzü, Dünya'nın en büyük beyaz yıldızı olup...`<br>*(✅ EOS — 23 token)* | Film başlığı kirliliği gitti; fakat 85M modelin olgusal renk bilgisi halüsinasyona açıktır. |
| **C05**<br>*"Türkiye hangi kıtada yer alır?"* | `Türkiye hangi kıtada yer alır? Kaynakça Ayrıca bakınız...`<br>*(❌ 128 tok kesildi)* | `Yunanistan`<br>*(✅ EOS — 1 token — Hatalı)* | `Avrupa`<br>*(✅ EOS — 1 token — DOĞRU!)* | 🟢 **Tam isabet.** Pilot sürümde ülke adı söylerken Aşama 7'de doğru kıtayı bildi! |
| **D03**<br>*"Bana 3 meyve say."* | `Bana 3 meyve say. 1. Elma 2. Armut colspan=2 style=...`<br>*(❌ 128 tok kesildi)* | `Bana 3 meyve say.`<br>*(✅ EOS — 5 token)* | `Bana 3 meyve say.`<br>*(✅ EOS — 5 token)* | Liste ve sayı kısıtlamalı emirlerde model soruyu kullanıcıya yankılamaktadır (Echoing). |
| **D20**<br>*"Bana kısa bir teşekkür cümlesi yaz."* | `Bana kısa bir teşekkür cümlesi yaz. Teşekkürler (albüm)...`<br>*(❌ 128 tok kesildi)* | `Bana kısa bir teşekkür.`<br>*(✅ EOS — 5 token)* | `"Bu bizim beni unutma."`<br>*(✅ EOS — 7 token)* | Albüm dökümü yerine doğrudan tırnak içinde duygusal bir cümle kurmayı başardı. |
| **E01**<br>*"Şu metni özetle: Türkiye Asya ve Avrupa..."* | `Toprakları bulunan bir ülkedir. Başkenti Ankara... Kaynakça...`<br>*(❌ 128 tok kesildi)* | `2,6 milyon`<br>*(✅ EOS — 4 token)* | `Dünya, dünya, Avrupa, Asya...`<br>*(✅ EOS — 32 token)* | Metin içindeki kıtaları yakalayarak özet çıkarmaya çalıştı, durma oranı %100 oldu. |
| **E05**<br>*"Şu metni özetle: Sağlıklı beslenmek..."* | `Sağlıklı beslenmek için sebze meyve... class="wikitable"...`<br>*(❌ 128 tok kesildi)* | `Çoklıklı beslenmek için süt ürünleri, meyve...`<br>*(❌ 128 tok kesildi)* | `Bir içmezlik için süt ve yağ arasında bir bağ oluşturur.`<br>*(✅ EOS — 13 token)* | 🟢 Pilotta 128 token takılıp duramazken 750 adımda **13 tokenda net özet çıkarıp durdu.** |
| **F05**<br>*"Konuyu değiştirme: Elma hakkında bir şey söyle."* | `Elma (anlam ayrımı) Elma, gülgiller familyasından...`<br>*(❌ 128 tok kesildi)* | `Mavi ve yeşil bir renk.`<br>*(✅ EOS — 6 token)* | `Mavi ve yeşil bir renk.`<br>*(✅ EOS — 6 token)* | Ansiklopedi ayrım sayfasını bırakıp elmanın rengini söyleyerek konuyu korudu ve durdu. |

---

## 🎯 Benchmark Test Seti Detayı

Test seti (`TESTLER/benchmark_questions.json`) **6 temel kategoride** dengelenmiş **150 Türkçe sorudan** oluşmaktadır:

| Kategori Kodu | Kategori Adı | Soru Sayısı | Değerlendirilen Beceri |
|:---:|---|:---:|---|
| **A** | Selamlaşma ve Rol Tanıma | 25 | Selamlamaya uygun yanıt, kibarlık, EOS ile durma |
| **B** | Türkçe Cümle Tamamlama | 30 | Dilbilgisi, ek uyumu, Türkçe söz dizimi |
| **C** | Basit Soru-Cevap (Faktüel) | 25 | Temel genel kültür, doğru olgu aktarımı, halüsinasyon kontrolü |
| **D** | Talimat Takibi | 25 | Verilen adede/şarta uyma, maddeleme |
| **E** | Kısa Özetleme | 25 | Sebep-sonuç ilişkisi, basit çıkarımlar |
| **F** | Tekrar ve Tutarlılık | 20 | Basit tekrar ve konu değiştirme kontrolü |

**Değerlendirme Koşulları:** `temperature=0.7`, `top_k=50`, `max_tokens=128`, `seed=42`, `device=cpu`

**Değerlendirme Dosyaları:**
| Dosya | Açıklama |
|---|---|
| `TESTLER/eval_v1_baseline.json` | v1 Base modelin 150 soruluk ham çıktıları |
| `TESTLER/eval_v2_sft.json` | v2 SFT Pilot (250 adım) model çıktıları |
| `TESTLER/eval_v2_stage7.json` | v2 SFT Aşama 7 (750 adım) final model çıktıları |
| `TESTLER/eval_v*_summary.txt` | Her modelin kategori bazlı özet metrikleri |

---

## 📁 Depo ve Klasör Yapısı

Proje karmaşayı önlemek amacıyla modüler bir mimariyle organize edilmiştir:

```text
Cores-AI-TR-v1.0.0-85M/
├── model.py                      # 🧠 NanoGPT tabanlı 81M Transformer model mimarisi
├── configurator.py               # ⚙️ Komut satırı argüman override mekanizması
├── tokenizer_tr_v2.json          # 📖 50.000 kelimelik özel Türkçe BPE Tokenizer
├── requirements.txt              # 📦 Python bağımlılıkları (PyTorch, Tokenizers, Matplotlib vb.)
│
├── 💬 ÇIKARIM VE SOHBET (INFERENCE)
│   ├── sample_sft.py             # 🌟 v2 SFT İnteraktif Terminal Sohbet Konsolu (Çok turlu)
│   └── sample_tr.py              # 📝 v1 Ham metin tamamlama betiği
│
├── 🏋️ EĞİTİM MOTORLARI (TRAINING ENGINES)
│   ├── train_sft.py              # 🚀 v2 SFT Eğitim Motoru (Loss masking, cosine decay, eval)
│   ├── data_sft.py               # 🔄 SFT PyTorch DataLoader (Dinamik padding, -1 maskeleme)
│   ├── prepare_sft.py            # 🧹 SFT Veri Temizleme & Tokenizasyon (8.936 örnek)
│   ├── train.py                  # ⚙️ v1 Pre-training Ana Eğitim Motoru
│   └── prepare_tr.py             # 📄 v1 Ön eğitim veri hazırlama betiği
│
├── 🔬 DEĞERLENDİRME MOTORLARI (EVALUATION)
│   ├── evaluate_sft.py           # 🔬 Güvenli SFT Benchmark Değerlendirme Betiği
│   └── evaluate.py               # 🔬 v1 Baseline Değerlendirme Betiği
│
├── 📦 MODEL CHECKPOINTLERİ (Hugging Face'de barınır; .gitignore ile hariç tutulur)
│   ├── out-cores-85m-sft-v2/     # 🟢 RESMİ v2 SFT MODELİ (750 Adım, Val: 3.9976)
│   │   ├── ckpt.pt               #    Standart ağırlık dosyası (978 MB)
│   │   └── ckpt_best.pt          #    En iyi kontrol noktası yedeği
│   └── out-cores-85m-test/       # 🔒 ORİJİNAL v1 BASE MODEL (5.000 Adım, Val: 2.8424)
│       └── ckpt.pt               #    Salt-okunur korunan ham model ağırlığı
│
├── 📊 TEST VE BENCHMARK SÜİTİ (TESTLER/)
│   ├── benchmark_questions.json  # 📋 150 Soruluk Sabit Test Seti (Korunan, sızıntısız)
│   ├── eval_v1_baseline.json     # 📊 v1 Baseline modelin 150 soruluk çıktıları
│   ├── eval_v2_sft.json          # 📊 v2 Pilot (250 adım) modelin çıktıları
│   ├── eval_v2_stage7.json       # 📊 v2 Aşama 7 (750 adım) final model çıktıları
│   ├── Cores_AI_TR_v2_Durum_Raporu.pdf # 📑 Yüksek çözünürlüklü detaylı grafikli PDF durum raporu
│   ├── cores_v2_durum_ve_plan.md # 🗺️ Aşama aşama teknik yol haritası ve notlar
│   ├── charts/                   # 📈 Üretilen yüksek çözünürlüklü grafikler (Loss, EOS vb.)
│   ├── temp_sft_pilot/           # 💾 Pilot (250 adım) kontrol noktası yedeği
│   ├── temp_sft_stage7/          # 💾 Aşama 7 (750 adım) kontrol noktası yedeği
│   └── legacy_logs/              # 📜 Tarihsel ön eğitim ve prompt günlükleri
│
├── 🛠️ ANALİZ VE DENETİM ARAÇLARI (tools/)
│   ├── compare_3way.py           # 📈 3 Yönlü Karşılaştırmalı Metrik Analiz Aracı
│   ├── compare_eval.py           # 📊 İki Yönlü Model Karşılaştırma Betiği
│   ├── detailed_analysis.py      # 🔍 Kategori Bazlı Detaylı Kalite İncelemesi
│   ├── generate_pdf_report.py    # 🖨️ PDF Durum Raporu Derleyici
│   ├── audit_data_quality.py     # 🧹 Veri Kalite ve Temsili Örnek Denetimi
│   └── test_sft_batch.py         # 🧪 SFT Batch ve Kontrol Noktası Bütünlük Testi
│
├── data/
│   └── sft_pilot/                # 📂 İşlenmiş SFT Veri Kümesi (train.jsonl, val.jsonl)
│       └── dataset_stats.json    #    Detaylı kaynak ve token istatistikleri
│
└── config/
    └── train_cores_85m_test.py   # ⚙️ v1 Ön Eğitim Konfigürasyonu (Colab T4 için optimize)
```

---

## 🛠️ Analiz ve Denetim Araçları (tools/)

Tüm analiz, karşılaştırma ve raporlama araçları `tools/` dizininde toplanmıştır. Proje kök dizininden çalıştırılmalıdır:

| Araç | Açıklama | Komut |
|---|---|---|
| `compare_3way.py` | v1 Base, v2 Pilot ve v2 Aşama 7 modellerinin 150 soruluk benchmark sonuçlarını SHA-256 doğrulamalı karşılaştırır. EOS, Vikipedi kirliliği, tekrar oranları ve kategori tablosu üretir. | `python tools/compare_3way.py` |
| `compare_eval.py` | v1 vs v2 iki yönlü karşılaştırma ve +/- fark analizi yapar. | `python tools/compare_eval.py` |
| `detailed_analysis.py` | A, C, D kategorilerindeki soru-cevap çiftlerini satır satır inceler. | `python tools/detailed_analysis.py` |
| `generate_pdf_report.py` | `TESTLER/charts/` grafiklerini kullanarak Microsoft Edge headless ile profesyonel PDF durum raporunu derler. | `python tools/generate_pdf_report.py` |
| `audit_data_quality.py` | `data/sft_pilot/train.jsonl` dosyasından kaynak bazlı 15 rastgele diyaloğu terminale basar. | `python tools/audit_data_quality.py` |
| `test_sft_batch.py` | Tek batch forward/backward gradyan testi, padding maskeleme doğrulaması ve checkpoint bütünlük kontrolü yapar. | `python tools/test_sft_batch.py` |
| `upload_to_hf.py` | 978 MB'lık model ağırlıklarını ve tokenizer'ı doğrudan Hugging Face Model Hub'a yükler. | `python tools/upload_to_hf.py --token <HF_TOKEN>` |

---

## 🏋️ Kendi Modelinizi Eğitme & Geliştirme

### 1. SFT Verisini Hazırlama
Açık lisanslı veri kümelerinden temiz SFT verisi oluşturmak için:
```bash
python prepare_sft.py --mode pilot
```

### 2. SFT Eğitimini Koşturma
Modeli diyalog formatında eğitmek için:
```bash
python train_sft.py \
    --init_ckpt out-cores-85m-test/ckpt.pt \
    --out_dir out-cores-85m-sft-v2 \
    --data_dir data/sft_pilot \
    --max_iters 500 \
    --eval_interval 50 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_iters 30 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --device cpu
```

### 3. Modeli Benchmark Üzerinde Değerlendirme
```bash
python evaluate_sft.py \
    --ckpt_path out-cores-85m-sft-v2/ckpt.pt \
    --output TESTLER/yeni_degerlendirme.json \
    --benchmark TESTLER/benchmark_questions.json \
    --device cpu
```

### 4. Sonuçları Karşılaştırma & Raporlama
```bash
# 3 nesil model karşılaştırma tablosunu görüntüleme
python tools/compare_3way.py

# Detaylı grafikli PDF durum raporunu yeniden derleme
python tools/generate_pdf_report.py
```

---

## 🚢 GitHub & Hugging Face Dağıtım Rehberi

### Sorun: GitHub'ın 100 MB Dosya Sınırı
GitHub, tekil dosyaların 100 MB'ı aşmasına izin vermez. Model checkpoint dosyaları (~978 MB) bu nedenle doğrudan GitHub'a yüklenemez.

### Çözüm: Kod → GitHub, Ağırlıklar → Hugging Face

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│          GITHUB REPO             │     │        HUGGING FACE HUB         │
│                                  │     │                                 │
│  ✅ model.py                     │     │  ✅ out-cores-85m-sft-v2/       │
│  ✅ train.py, train_sft.py       │     │     └── ckpt.pt (978 MB)        │
│  ✅ sample_sft.py                │     │     └── ckpt_best.pt (978 MB)   │
│  ✅ evaluate_sft.py              │     │  ✅ out-cores-85m-test/          │
│  ✅ tokenizer_tr_v2.json         │     │     └── ckpt.pt (978 MB)        │
│  ✅ TESTLER/ (JSON, TXT, PNG)    │     │  ✅ tokenizer_tr_v2.json        │
│  ✅ tools/                       │     │  ✅ README.md (Model Card)      │
│  ✅ data/sft_pilot/              │     │                                 │
│  ✅ config/                      │     │                                 │
│  ❌ out-*/ (.gitignore ile)      │     │                                 │
│  ❌ .venv/                       │     │                                 │
└──────────────────────────────────┘     └──────────────────────────────────┘
```

### Adım 1: GitHub'a Kod Yükleme

```powershell
# 1. Git kurulu değilse: https://git-scm.com/download/win adresinden kurun

# 2. Proje dizininde terminal açın
cd C:\Users\abdur\Desktop\Cores-AI-TR-v1.0.0-85M-main

# 3. Git repository başlatın
git init
git add .
git commit -m "Cores-AI TR v1.0.0 Base + v2.0.0 SFT - İlk sürüm"

# 4. GitHub'da boş bir repo oluşturup bağlayın
git remote add origin https://github.com/KULLANICI_ADINIZ/Cores-AI-TR-v1.0.0-85M.git
git branch -M main
git push -u origin main
```

> [!NOTE]
> `.gitignore` dosyası `*.pt`, `out-*/`, `data/` ve `.venv/` dizinlerini otomatik olarak hariç tutar. GitHub'a büyük dosya yüklenmez.

### Adım 2: Hugging Face'e Model Ağırlıklarını Yükleme

#### Yöntem A: Hazır Python Yükleme Aracı ile (Önerilen - En Kolay)
Projede yer alan `tools/upload_to_hf.py` betiği, checkpoint ve tokenizer dosyalarını tek komutla Hugging Face Model Hub'a aktarır:

```powershell
python tools/upload_to_hf.py --token <HF_WRITE_TOKEN> --repo aapo33/cores-ai-tr-v1.0.0-85m
```
*(Token almak için: https://huggingface.co/settings/tokens adresinden "Write" rolünde bir Access Token oluşturun.)*

#### Yöntem B: Hugging Face CLI ile

```powershell
# 1. Hugging Face CLI ile giriş yapın
huggingface-cli login

# 2. Dosyaları yükleyin
huggingface-cli upload aapo33/cores-ai-tr-v1.0.0-85m out-cores-85m-sft-v2/ckpt.pt out-cores-85m-sft-v2/ckpt.pt
huggingface-cli upload aapo33/cores-ai-tr-v1.0.0-85m out-cores-85m-sft-v2/ckpt_best.pt out-cores-85m-sft-v2/ckpt_best.pt
huggingface-cli upload aapo33/cores-ai-tr-v1.0.0-85m out-cores-85m-test/ckpt.pt out-cores-85m-test/ckpt.pt
huggingface-cli upload aapo33/cores-ai-tr-v1.0.0-85m tokenizer_tr_v2.json tokenizer_tr_v2.json
```

### Adım 3: .gitignore Doğrulaması
Mevcut `.gitignore` dosyanız zaten doğru yapılandırılmıştır:
- `*.pt` → Tüm checkpoint dosyaları hariç
- `out-*/` → Tüm model çıktı dizinleri hariç
- `data/` → Eğitim verileri hariç (opsiyonel olarak eklenebilir)
- `.venv/` → Sanal ortam hariç

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında paylaşılmaktadır. Hem ticari hem de kişisel projelerinizde serbestçe kullanabilir, modeli geliştirebilir ve kendi veri kümelerinizle yeniden eğitebilirsiniz.
