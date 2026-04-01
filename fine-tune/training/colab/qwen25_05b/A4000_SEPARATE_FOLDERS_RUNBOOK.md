# A4000 Separate Folders Runbook

Bu surumde iki veri seti icin tamamen ayri calisma klasoru kullanilir.

Boylece su klasorler birbirinden tamamen ayrilir:

- `data/`
- `saves/`
- `trainer_log.jsonl`
- grafikler
- checkpoint'ler

Kullanilacak iki ayri klasor:

- `~/Desktop/qwen25_action_50_000_run`
- `~/Desktop/qwen25_ep_3500_run`

## Ortak not

Ayni conda environment kullanilabilir:

```bash
conda activate qwen25_fullft
```

Ama egitimleri ayni anda baslatma. Sirayla calistir.

---

## 1. Balanced veri seti: `Ali2023kosemen/action_50_000`

### 1.1 Klasoru olustur

```bash
mkdir -p ~/Desktop/qwen25_action_50_000_run
cd ~/Desktop/qwen25_action_50_000_run
```

### 1.2 LLaMA-Factory klonla

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### 1.3 Gerekli dosyalari buraya kopyala

Bu repodan su dosyalari bu klasore kopyala:

- `prepare_lunarlander_dataset_for_llamafactory.py`
- `run_full_ft_a4000_action_50_000_torchrun.sh`
- `plot_training_loss.py`
- `watch_training_metrics.py`
- `simple_lunarlander_inference_test.py`
- `archive_finetuned_model.py`

### 1.4 Dataset'i hazirla

```bash
python prepare_lunarlander_dataset_for_llamafactory.py \
  --dataset-id Ali2023kosemen/action_50_000 \
  --dataset-slug action_50_000
```

### 1.5 Egitimi baslat

```bash
chmod +x run_full_ft_a4000_action_50_000_torchrun.sh
./run_full_ft_a4000_action_50_000_torchrun.sh
```

### 1.6 Izleme

Ayri terminal:

```bash
conda activate qwen25_fullft
cd ~/Desktop/qwen25_action_50_000_run/LLaMA-Factory
python watch_training_metrics.py \
  --output-dir saves/qwen25_05b_base_full_ft_action_50_000_a4000
```

### 1.7 Test

```bash
python simple_lunarlander_inference_test.py \
  --model-dir saves/qwen25_05b_base_full_ft_action_50_000_a4000 \
  --test-file data/action_50_000_eval.json \
  --show-examples 5
```

### 1.8 Grafik

```bash
python plot_training_loss.py \
  --output-dir saves/qwen25_05b_base_full_ft_action_50_000_a4000
```

---

## 2. Unbalanced veri seti: `Ali2023kosemen/ep_3500`

### 2.1 Klasoru olustur

```bash
mkdir -p ~/Desktop/qwen25_ep_3500_run
cd ~/Desktop/qwen25_ep_3500_run
```

### 2.2 LLaMA-Factory klonla

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### 2.3 Gerekli dosyalari buraya kopyala

Bu repodan su dosyalari bu klasore kopyala:

- `prepare_lunarlander_dataset_for_llamafactory.py`
- `run_full_ft_a4000_ep_3500_torchrun.sh`
- `plot_training_loss.py`
- `watch_training_metrics.py`
- `simple_lunarlander_inference_test.py`
- `archive_finetuned_model.py`

### 2.4 Dataset'i hazirla

```bash
python prepare_lunarlander_dataset_for_llamafactory.py \
  --dataset-id Ali2023kosemen/ep_3500 \
  --dataset-slug ep_3500
```

### 2.5 Egitimi baslat

```bash
chmod +x run_full_ft_a4000_ep_3500_torchrun.sh
./run_full_ft_a4000_ep_3500_torchrun.sh
```

### 2.6 Izleme

Ayri terminal:

```bash
conda activate qwen25_fullft
cd ~/Desktop/qwen25_ep_3500_run/LLaMA-Factory
python watch_training_metrics.py \
  --output-dir saves/qwen25_05b_base_full_ft_ep_3500_a4000
```

### 2.7 Test

```bash
python simple_lunarlander_inference_test.py \
  --model-dir saves/qwen25_05b_base_full_ft_ep_3500_a4000 \
  --test-file data/ep_3500_eval.json \
  --show-examples 5
```

### 2.8 Grafik

```bash
python plot_training_loss.py \
  --output-dir saves/qwen25_05b_base_full_ft_ep_3500_a4000
```

---

## Onerilen sira

1. once `action_50_000`
2. sonra `ep_3500`

Bu sekilde:

- klasorler karismaz
- dataset dosyalari karismaz
- checkpoint'ler karismaz
- karsilastirma yapmak kolay olur
