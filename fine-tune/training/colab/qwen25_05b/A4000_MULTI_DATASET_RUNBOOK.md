# A4000 Multi-Dataset Runbook

Bu not, A4000 uzerinde ayni ortamla iki farkli veri setini ayri ayri fine-tune etmek icin hazirlandi:

- `Ali2023kosemen/action_50_000` -> balanced
- `Ali2023kosemen/ep_3500` -> unbalanced

Her ikisi de ayni model ve ayni hyperparameter ile calisir. Fark sadece:

- hazirlanan train/eval dosyalari
- LLaMA-Factory dataset anahtarlari
- `output_dir`

## Server'a kopyalanacak dosyalar

`LLaMA-Factory/` icine su dosyalari kopyala:

- `prepare_lunarlander_dataset_for_llamafactory.py`
- `run_full_ft_a4000_action_50_000_torchrun.sh`
- `run_full_ft_a4000_ep_3500_torchrun.sh`
- `plot_training_loss.py`
- `watch_training_metrics.py`
- `simple_lunarlander_inference_test.py`
- `archive_finetuned_model.py`

## Ortak baslangic

```bash
conda activate qwen25_fullft
cd ~/Desktop/qwen25_fullft_run/LLaMA-Factory
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 1. Balanced veri seti: `Ali2023kosemen/action_50_000`

### Dataset hazirlama

```bash
python prepare_lunarlander_dataset_for_llamafactory.py \
  --dataset-id Ali2023kosemen/action_50_000 \
  --dataset-slug action_50_000
```

Bu komut su dosyalari uretir:

- `data/action_50_000_train.json`
- `data/action_50_000_eval.json`
- `data/action_50_000_full.json`

ve `dataset_info.json` icine su anahtarlari ekler:

- `action_50_000_train`
- `action_50_000_eval`
- `action_50_000_full`

### Egitim

```bash
chmod +x run_full_ft_a4000_action_50_000_torchrun.sh
./run_full_ft_a4000_action_50_000_torchrun.sh
```

### Izleme

```bash
python watch_training_metrics.py \
  --output-dir saves/qwen25_05b_base_full_ft_action_50_000_a4000
```

### Egitim sonrasi test

```bash
python simple_lunarlander_inference_test.py \
  --model-dir saves/qwen25_05b_base_full_ft_action_50_000_a4000 \
  --test-file data/action_50_000_eval.json \
  --show-examples 5
```

### Grafik

```bash
python plot_training_loss.py \
  --output-dir saves/qwen25_05b_base_full_ft_action_50_000_a4000
```

## 2. Unbalanced veri seti: `Ali2023kosemen/ep_3500`

### Dataset hazirlama

```bash
python prepare_lunarlander_dataset_for_llamafactory.py \
  --dataset-id Ali2023kosemen/ep_3500 \
  --dataset-slug ep_3500
```

Bu komut su dosyalari uretir:

- `data/ep_3500_train.json`
- `data/ep_3500_eval.json`
- `data/ep_3500_full.json`

ve `dataset_info.json` icine su anahtarlari ekler:

- `ep_3500_train`
- `ep_3500_eval`
- `ep_3500_full`

### Egitim

```bash
chmod +x run_full_ft_a4000_ep_3500_torchrun.sh
./run_full_ft_a4000_ep_3500_torchrun.sh
```

### Izleme

```bash
python watch_training_metrics.py \
  --output-dir saves/qwen25_05b_base_full_ft_ep_3500_a4000
```

### Egitim sonrasi test

```bash
python simple_lunarlander_inference_test.py \
  --model-dir saves/qwen25_05b_base_full_ft_ep_3500_a4000 \
  --test-file data/ep_3500_eval.json \
  --show-examples 5
```

### Grafik

```bash
python plot_training_loss.py \
  --output-dir saves/qwen25_05b_base_full_ft_ep_3500_a4000
```

## Onemli not

- Ayni A4000 uzerinde bu iki egitimi ayni anda baslatma.
- Sira ile calistir:
  1. `action_50_000`
  2. `ep_3500`
- Her egitim kendi `output_dir` klasorune yazdigi icin birbirinin ustune yazmaz.
