# Prompt Injection Defense

Prompt injection saldırılarına karşı iki katmanlı bir savunma mimarisinin değerlendirme repodur.
Temel amacı, **DefensiveToken** ve **Bi-LSTM cascade** tekniklerini `Qwen/Qwen2.5-7B-Instruct` üzerinde birlikte test etmek ve üç savunma modunu (baseline, defense, bilstm-defense) karşılaştırılabilir metriklerle ölçmektir.

---

## Upstream Repoların Rolü

Bu repo iki yayınlanmış çalışmanın üzerine inşa edilmiştir:

| Repo | Kullanım Biçimi |
|------|----------------|
| [Sizhe-Chen/DefensiveToken](https://github.com/Sizhe-Chen/DefensiveToken) | Defensive token embedding'leri ve chat template'i doğrudan kullanılmaktadır. `src/official_stacks/defensivetoken/` altında Qwen'e uyarlanmış haliyle bulunur. |
| [facebookresearch/Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign) | AlpacaFarm değerlendirme hattı ve GCG saldırı kodu `gcg_vendor/` olarak vendorlanmıştır. Evaluation pipeline, Qwen ile çalışacak şekilde yeniden yazılmıştır. |

---

## Bu Reponun Katkısı

Upstream repoların her ikisi de birbirinden bağımsız savunmalar önerir. Bu repo şunu ekler:

**Bi-LSTM Cascade Defense** — LLM'e giden her isteği önce hafif bir Bi-LSTM sınıflandırıcısından geçirir. Saldırı tespit edilirse istek bloklanır ve LLM hiç çalışmaz; temiz istekler savunmalı modele iletilir.

```
İstek → [Bi-LSTM ön filtre]
              ├─ Injection tespit → Blokla (sabit güvenli yanıt)
              └─ Temiz          → [DefensiveToken modeli] → Yanıt
```

Bu kademeli yapı, ASR'ı sıfıra yaklaştırırken win-rate'i (utility) korur.

---

## Değerlendirilen Modlar

| Mod | Ne yapar | Savunma |
|-----|----------|---------|
| `baseline` | Ham `Qwen/Qwen2.5-7B-Instruct` | Yok |
| `defense` | DefensiveToken embedding'li Qwen | Defensive tokens |
| `bilstm-defense` | Bi-LSTM ön filtre + DefensiveToken modeli | Bi-LSTM + Defensive tokens |

### Metrikler

- **Win-Rate** — AlpacaFarm utility testi; GPT-4 Turbo referans çıktılarına karşı `gpt-4o-mini` judge ile ölçülür. Yüksek = iyi.
- **ASR** (Attack Success Rate) — Üç prompt injection varyantına karşı başarı oranı (`ignore`, `completion`, `completion_ignore`). Düşük = iyi.
- **GCG-ASR** — Gradyan tabanlı optimize edilmiş adaptive saldırıya karşı ASR. Düşük = iyi.
- **Bi-LSTM OOD** — `deepset/prompt-injections` test split'ine karşı Precision / Recall / F1 / FPR.

---

## Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
pip install datasets pyarrow
```

GPU önerilir (Qwen 7B için en az 16 GB VRAM). Google Colab T4 ile çalışır.

### 1. AlpacaFarm verisini indir

```bash
python scripts/bootstrap_qwen_alpaca_data.py
```

### 2. OpenAI API yapılandırması

`src/official_stacks/meta_secalign/data/openai_configs.yaml` dosyasını oluştur:

```yaml
default:
  - client_class: "openai.OpenAI"
    api_key: "YOUR_OPENAI_API_KEY"
    model: "gpt-4o-mini"
    min_interval_seconds: 1.5
    max_retries: 8
    backoff_seconds: 10.0
```

### 3. DefensiveToken modelini hazırla (~10-15 dk)

```bash
python src/model/setup.py
```

Çıktı: `src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens/`

---

## Bi-LSTM Dedektörü (bilstm-defense için)

### Eğitim

```bash
python scripts/train_bilstm_detector.py \
    --data src/official_stacks/meta_secalign/data/davinci_003_outputs.json \
    --output bilstm_checkpoint.pt \
    --epochs 20 \
    --rogue-security data/prompt-injections-benchmark/test.csv \
    --hackaprompt-hf \
    --hf-token YOUR_HF_TOKEN
```

Eğitim verisi kaynaklarının tamamı:

| Kaynak | Tip |
|--------|-----|
| AlpacaFarm `davinci_003_outputs.json` (train split) | Negatif (temiz) + Pozitif (saldırılı) |
| `deepset/prompt-injections` train split | Genel injection örnekleri |
| `rogue-security/prompt-injections-benchmark` | Ek injection örnekleri |
| HackAPrompt (HF Hub, gated) | Gerçek dünya yarışma injection'ları |

Bi-LSTM, AlpacaFarm'ın train split'inde eğitilir; test split'e dokunulmaz — OOD değerlendirmesi için ayrılır.

### OOD Değerlendirme

```bash
python scripts/eval_bilstm_ood.py \
    --checkpoint bilstm_checkpoint.pt \
    --output docs/raporlar/bilstm_ood.json \
    --threshold 0.3
```

---

## Değerlendirme

Her mod bağımsız cache'e yazar; birinin sonucu diğerini etkilemez.

### Baseline

```bash
python scripts/run_qwen_alpaca_eval.py --mode baseline --skip-gcg
```

### Defense (DefensiveTokens)

```bash
python scripts/run_qwen_alpaca_eval.py --mode defense --skip-gcg
```

### Bi-LSTM Cascade Defense

```bash
python scripts/run_qwen_alpaca_eval.py \
    --mode bilstm-defense \
    --bilstm-checkpoint bilstm_checkpoint.pt \
    --skip-gcg
```

### GCG Adaptive Saldırı (opsiyonel, ~15-20 dk)

```bash
python scripts/run_qwen_alpaca_eval.py \
    --mode defense \
    --gcg-max-samples 50
```

`--skip-gcg` olmadan çalıştırıldığında GCG otomatik dahil edilir. `--gcg-max-samples` ile örnek sayısı sınırlanabilir.

---

## Sonuç Karşılaştırması

```python
import json
from pathlib import Path

REPORT_DIR = Path("docs/raporlar/qwen_alpaca")
print(f"{'Mod':<22} {'Win-Rate':>10} {'ASR':>10} {'GCG-ASR':>10}")
print("-" * 56)
for mode in ["baseline", "defense", "bilstm-defense"]:
    p = REPORT_DIR / f"{mode}.json"
    if not p.exists():
        print(f"{mode:<22} (henüz çalıştırılmadı)")
        continue
    m = json.loads(p.read_text())["metrics"]
    gcg = f"{m['gcg_asr']:.4f}" if "gcg_asr" in m else "   -"
    print(f"{mode:<22} {m['win_rate']:>9.4f}  {m['asr']:>9.4f}  {gcg:>9}")
```

Raporlar `docs/raporlar/qwen_alpaca/{mode}.json` altında saklanır.

---

## Colab

Tüm adımları içeren notebook: [`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb)

Runtime olarak **T4 GPU** seç, ardından hücreleri sırasıyla çalıştır.

---

## Proje Yapısı

```
prompt-injection-defense/
├── scripts/
│   ├── bootstrap_qwen_alpaca_data.py   # AlpacaFarm veri indirme
│   ├── train_bilstm_detector.py        # Bi-LSTM eğitimi
│   ├── eval_bilstm_ood.py              # OOD değerlendirme
│   └── run_qwen_alpaca_eval.py         # Ana değerlendirme giriş noktası
├── src/
│   ├── bilstm/                         # Bi-LSTM model ve tokenizer
│   ├── model/setup.py                  # DefensiveToken model hazırlama
│   └── official_stacks/
│       ├── defensivetoken/             # DefensiveToken entegrasyonu
│       └── meta_secalign/              # AlpacaFarm eval + GCG vendor
├── docs/raporlar/                      # Değerlendirme çıktıları
└── notebooks/colab_runbook.ipynb       # Adım adım Colab rehberi
```
