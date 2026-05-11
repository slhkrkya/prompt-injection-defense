# Colab Runbook

Bu rehber, projeyi Google Colab notebook hucrelerinde dogrudan calistirmak icin hazirlanmistir.

## Varsayimlar

- Repo Colab icinde `/content/prompt-injection-defense` altina klonlanacak.
- Klonlama `defensive_position` branch'i uzerinden yapilacak.
- Hedef model yalnizca `Qwen/Qwen2.5-7B-Instruct`.
- Judge modeli yalnizca `gpt-4o-mini` (AlpacaEval2 protokolu; referans model GPT-4 Turbo, 805 ornek).
- Calistirilacak deneyler: `baseline`, `defense` ve token pozisyon ablation (`prefix`, `suffix`, `sandwich`, `per_user`).

## 1. Repo'yu Klonla

```python
%cd /content
!git clone --branch defensive_position https://github.com/ABerkeBilgin/prompt-injection-defense.git prompt-injection-defense
%cd /content/prompt-injection-defense
!pwd
```

Repo zaten varsa:

```python
%cd /content/prompt-injection-defense
!git branch --show-current
!pwd
```

## 2. Paketleri Kur

```python
%cd /content/prompt-injection-defense
!pip install -r requirements.txt
```

## 3. OpenAI Config Dosyasini Doldur

Dosya yolu:

- `/content/prompt-injection-defense/src/official_stacks/meta_secalign/data/openai_configs.yaml`

Mevcut dosyayi gormek icin:

```python
%cd /content/prompt-injection-defense
!cat src/official_stacks/meta_secalign/data/openai_configs.yaml
```

Gercek OpenAI anahtariniz ile guncellemek icin:

```python
%%writefile /content/prompt-injection-defense/src/official_stacks/meta_secalign/data/openai_configs.yaml
default:
  - client_class: "openai.OpenAI"
    api_key: "YOUR_OPENAI_API_KEY"
    model: "gpt-4o-mini"
    min_interval_seconds: 1.5
    max_retries: 8
    backoff_seconds: 10.0
```

## 4. Alpaca Verisini Indir

```python
%cd /content/prompt-injection-defense
!python scripts/bootstrap_qwen_alpaca_data.py
```

Olusacak dosya:

- `/content/prompt-injection-defense/src/official_stacks/meta_secalign/data/davinci_003_outputs.json`

Kontrol:

```python
%cd /content/prompt-injection-defense
!ls -lh src/official_stacks/meta_secalign/data/
```

## 5. Dry-Run ile Yol Kontrolu

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode baseline --dry-run
!python scripts/run_qwen_alpaca_eval.py --mode defense --dry-run
```

Beklenen noktalar:

- `model` degeri `Qwen/Qwen2.5-7B-Instruct`
- `judge_model` degeri `gpt-4o-mini` (AlpacaEval2 protokolu)
- `openai_config_path` dogru dosyayi gostermeli

## 6. Defended Modeli Hazirla

```python
%cd /content/prompt-injection-defense
!python src/model/setup.py
```

Beklenen klasor:

- `/content/prompt-injection-defense/src/official_stacks/defensivetoken/Qwen/Qwen2.5-7B-Instruct-5DefensiveTokens`

Kontrol:

```python
%cd /content/prompt-injection-defense
!ls -lh src/official_stacks/defensivetoken/Qwen/
```

## 7. Baseline Evaluation

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode baseline --skip-gcg
```

Baseline raporu:

- `/content/prompt-injection-defense/docs/raporlar/qwen_alpaca/baseline.json`

## 8. Defense Evaluation

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode defense --skip-gcg
```

Tam GCG dahil calistirmak icin:

```python
%cd /content/prompt-injection-defense
!python scripts/run_qwen_alpaca_eval.py --mode defense
```

Defense raporu:

- `/content/prompt-injection-defense/docs/raporlar/qwen_alpaca/defense.json`

## 9. Sonuclari Gor

Tam JSON ciktilarini okumak icin:

```python
%cd /content/prompt-injection-defense
!cat docs/raporlar/qwen_alpaca/baseline.json
!cat docs/raporlar/qwen_alpaca/defense.json
```

Sadece metrikleri hizlica gormek icin:

```python
%cd /content/prompt-injection-defense
!python - <<'PY'
import json
from pathlib import Path

for name in ["baseline", "defense"]:
    path = Path("docs/raporlar/qwen_alpaca") / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    print(name, data["metrics"])
PY
```

## 10. Token Pozisyon Ablation

n=5 token sabit, 4 farkli pozisyon karsilastirilir: `prefix` (paper baseline), `suffix`, `sandwich`, `per_user`.

Her pozisyon icin ayri bir model diske kaydedilir (~14 GB/pozisyon). Oncesinde Adim 6'nin tamamlanmis olmasi gerekir.

Dry-run ile yapilandirilmayi kontrol et:

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py --dry-run
```

Tek pozisyon test et (once `suffix`'i dene):

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py --positions suffix
```

Tam grid (tum 4 pozisyon, GCG yok, ~8-16 saat GPU):

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py
```

Belirli pozisyonlari secmek icin:

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py --positions prefix suffix sandwich per_user
```

Ablation sonuclarini gor:

```python
%cd /content/prompt-injection-defense
!cat docs/raporlar/ablation/summary.csv
```

Beklenen `summary.csv` ciktisi:

```
position,win_rate,asr
prefix,0.XX,0.XX
suffix,0.XX,0.XX
sandwich,0.XX,0.XX
per_user,0.XX,0.XX
```

## Tek Parca Colab Blogu

```python
%cd /content
!git clone --branch defensive_position https://github.com/ABerkeBilgin/prompt-injection-defense.git prompt-injection-defense
%cd /content/prompt-injection-defense
!pip install -r requirements.txt

!cat > src/official_stacks/meta_secalign/data/openai_configs.yaml <<'YAML'
default:
  - client_class: "openai.OpenAI"
    api_key: "YOUR_OPENAI_API_KEY"
    model: "gpt-4o-mini"
    min_interval_seconds: 1.5
    max_retries: 8
    backoff_seconds: 10.0
YAML

%cd /content/prompt-injection-defense
!python scripts/bootstrap_qwen_alpaca_data.py
!python scripts/run_qwen_alpaca_eval.py --mode baseline --dry-run
!python scripts/run_qwen_alpaca_eval.py --mode defense --dry-run
!python src/model/setup.py
!python scripts/run_qwen_alpaca_eval.py --mode baseline --skip-gcg
!python scripts/run_qwen_alpaca_eval.py --mode defense --skip-gcg
!python scripts/run_ablation.py --dry-run
!python scripts/run_ablation.py
```

