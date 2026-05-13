# Colab Runbook

Bu rehber, projeyi Google Colab icinde yeni sandwich-tail ablation deneyleri icin calistirir.

## Varsayimlar

- Repo Colab icinde `/content/prompt-injection-defense` altina klonlanacak.
- Klonlama `defensive_position` branch'i uzerinden yapilacak.
- Hedef model yalnizca `Qwen/Qwen2.5-7B-Instruct`.
- Judge modeli `gpt-4o-mini`.
- GCG test edilmeyecek.
- Ablation grid: `prefix`, `sandwich_tail_1`, `sandwich_tail_2`, `sandwich_tail_3`, `sandwich_tail_5`.

## Deney Mantigi

`prefix` paper baseline'dir:

```text
[DT0][DT1][DT2][DT3][DT4]
system + user
assistant
```

Sandwich-tail varyantlari prefix'i sabit tutar, sadece assistant cevabindan hemen once kac defensive token eklenecegini degistirir:

```text
[DT0][DT1][DT2][DT3][DT4]
system + user
[tail tokens]
assistant
```

Varyantlar:

```text
sandwich_tail_1 -> assistant oncesi [DT4]
sandwich_tail_2 -> assistant oncesi [DT3][DT4]
sandwich_tail_3 -> assistant oncesi [DT2][DT3][DT4]
sandwich_tail_5 -> assistant oncesi [DT0][DT1][DT2][DT3][DT4]
```

Hedef: ASR prefix seviyesinde kalirken win_rate'i artiran en kucuk tail yapisini bulmak.

## 1. Repo'yu Klonla

```python
%cd /content
!git clone --branch defensive_positionv2 https://github.com/ABerkeBilgin/prompt-injection-defense.git prompt-injection-defense
%cd /content/prompt-injection-defense
!pwd
```

Repo zaten varsa:

```python
%cd /content/prompt-injection-defense
!git branch --show-current
!git pull
```

## 2. Paketleri Kur

```python
%cd /content/prompt-injection-defense
!pip install -r requirements.txt
```

## 3. OpenAI Config Dosyasini Doldur

Dosya yolu:

```text
/content/prompt-injection-defense/src/official_stacks/meta_secalign/data/openai_configs.yaml
```

Gercek OpenAI anahtariniz ile guncelleyin:

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

Beklenen dosya:

```text
/content/prompt-injection-defense/src/official_stacks/meta_secalign/data/davinci_003_outputs.json
```

## 5. Dry-Run ile Kontrol Et

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py --dry-run
```

Beklenen `positions` listesi:

```text
prefix
sandwich_tail_1
sandwich_tail_2
sandwich_tail_3
sandwich_tail_5
```

## 6. Hizli Tek Varyant Denemesi

Once en ucuz tail varyantini deneyin:

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py --positions sandwich_tail_1
```

Sonuc iyi gorunurse diger tail varyantlarini calistirin:

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py
```

## 7. Tam Ablation Grid

Tum grid'i calistirmak icin:

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py
```

Onceki cache'i silerek sifirdan calistirmak icin:

```python
%cd /content/prompt-injection-defense
!python scripts/run_ablation.py --force
```

## 8. Sonuclari Oku

Ozet CSV:

```python
%cd /content/prompt-injection-defense
!cat docs/raporlar/ablation/summary.csv
```

Beklenen kolonlar:

```text
position,learned_tokens,prefix_tokens,tail_tokens,insertion_sites,win_rate,asr
```

Sonuclari siralamak icin:

```python
%cd /content/prompt-injection-defense
!python - <<'PY'
import pandas as pd

df = pd.read_csv("docs/raporlar/ablation/summary.csv")
print(df.sort_values(["asr", "win_rate"], ascending=[True, False]).to_string(index=False))
PY
```

En iyi aday, ASR'si prefix ile ayni veya cok yakin olup win_rate'i en yuksek olan satirdir.

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

!python scripts/bootstrap_qwen_alpaca_data.py
!python scripts/run_ablation.py --dry-run
!python scripts/run_ablation.py --positions sandwich_tail_1
!python scripts/run_ablation.py
!cat docs/raporlar/ablation/summary.csv
```
