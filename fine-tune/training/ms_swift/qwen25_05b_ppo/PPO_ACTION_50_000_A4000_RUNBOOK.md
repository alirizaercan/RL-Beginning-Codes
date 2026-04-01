# Qwen2.5-0.5B PPO on A4000

Bu klasor tekrar gercek `PPO` akisi icin duzenlendi.

Amac:

- LLaMA-Factory ile yaptiginiz SFT asamasindan sonra
- ayni LunarLander gorevi icin
- ms-swift ile `PPO` calistirmak

Veri seti:

- `Ali2023kosemen/ppo_action_50_000`

Veri seti sutunlari:

- `messages`
- `solution`

Ornek:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "State: [x=0.1080, y=0.8034, vx=0.1094, vy=-0.8267, angle=0.2673, angular_vel=-0.0074, left_leg=0.0, right_leg=0.0]. What action should the lander take?"
    }
  ],
  "solution": "Action: 0"
}
```

## Onemli netlik

Bu klasor `PPO` icin hazirlandi.

Ama ms-swift'te gercek PPO akisi su sekilde olur:

1. elinizde bir baslangic modeli olur
2. `RM` yani reward model egitilir
3. sonra `PPO` policy egitilir

Bu yuzden burada iki training script var:

- `run_rm_action_50_000_a4000.sh`
- `run_ppo_action_50_000_a4000.sh`

Bu fazladan bir tercih degil.
ms-swift `PPO` akisinin parcasidir.

## Dataset neden hazirlaniyor?

Kaynak datasetiniz dogru:

- `messages`
- `solution`

Ama PPO icin iki ayri veri dosyasi gerekiyor:

1. `RM` stage
   - prompt
   - iyi cevap
   - kotu cevap

2. `PPO` stage
   - prompt/query

Bu yuzden `prepare_action_50_000_ppo_dataset.py` su dosyalari uretir:

- `action_50_000_rm_train.jsonl`
- `action_50_000_rm_eval.jsonl`
- `action_50_000_ppo_train.jsonl`
- `action_50_000_ppo_eval.jsonl`
- `action_50_000_ppo_eval.json`

Buradaki `rejected_response`, `solution` disinda farkli bir action secilerek RM icin uretilir.

## Accuracy burada nasil yorumlanir?

PPO training sirasinda LLaMA-Factory'deki gibi dogrudan `eval_accuracy` beklememek lazim.

Burada genelde sunlari gorursun:

- reward ile ilgili metricler
- policy/value loss metricleri
- KL metricleri
- learning rate

Final action accuracy icin ayri test scripti kullaniyoruz:

- `simple_lunarlander_ppo_inference_test.py`

Bu test scripti:

- modeli generate ile calistirir
- `Action: <id>` parse eder
- `solution` ile karsilastirir

Yani bu scriptteki `Generation action accuracy`:

- gercek inference accuracy'dir
- reward modeli degil, dogrudan final policy'yi test eder

## Template ayni mi?

Evet.

Training ve testte ayni Qwen chat template kullaniliyor:

- training: `--template qwen`
- test: `tokenizer.apply_chat_template(...)`

Yani:

- egitime nasil veriliyorsa
- testte de ayni template ailesi kullaniliyor

## A4000'de adim adim

### 1. Env olustur

```bash
conda create -n qwen25_ppo python=3.11 -y
conda activate qwen25_ppo
```

### 2. Yeni klasor olustur

```bash
mkdir -p ~/Desktop/qwen25_ppo_action_50_000_run
cd ~/Desktop/qwen25_ppo_action_50_000_run
```

### 3. ms-swift kur

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout release/3.12
pip install -e .
pip install datasets huggingface_hub pandas matplotlib
```

### 4. Bu dosyalari `ms-swift/` klasorune kopyala

- `lunarlander_rl_utils.py`
- `prepare_action_50_000_ppo_dataset.py`
- `run_rm_action_50_000_a4000.sh`
- `run_ppo_action_50_000_a4000.sh`
- `swift_metrics_utils.py`
- `watch_swift_training_metrics.py`
- `plot_swift_training_metrics.py`
- `simple_lunarlander_ppo_inference_test.py`
- `export_ppo_action_50_000_lora.sh`
- `archive_ppo_model.py`
- `fix_qwen_tokenizer_config.py`

### 5. Dataseti hazirla

```bash
python prepare_action_50_000_ppo_dataset.py
```

### 6. Scriptlere izin ver

```bash
chmod +x run_rm_action_50_000_a4000.sh
chmod +x run_ppo_action_50_000_a4000.sh
chmod +x export_ppo_action_50_000_lora.sh
```

### 6.5. Yerel SFT modelin tokenizer config dosyasini bir kere duzelt

LLaMA-Factory ile kaydedilmis bazi yerel Qwen modellerinde
`tokenizer_config.json` icindeki `extra_special_tokens` alani `[]` olarak geliyor.
Bu ms-swift/transformers kombinasyonunda hata cikariyor.

Bir kere su komutu calistir:

```bash
python fix_qwen_tokenizer_config.py \
  --model-dir /home/autodidactic/Desktop/qwen25_ep_3500_run/LLaMA-Factory/saves/qwen25_05b_base_full_ft_ep_3500_a4000
```

### 7. Reward model train

Burada `MODEL_PATH`, daha once SFT ile egittigin model olmali.
Yerel LLaMA-Factory klasoru kullaniyorsan `MODEL_TYPE=qwen2_5` vermen gerekir.

Ornek:

```bash
MODEL_TYPE=qwen2_5 \
MODEL_PATH=/home/autodidactic/Desktop/qwen25_ep_3500_run/LLaMA-Factory/saves/qwen25_05b_base_full_ft_ep_3500_a4000 \
./run_rm_action_50_000_a4000.sh
```

### 8. RM metriclerini izle

Ayri terminal:

```bash
conda activate qwen25_ppo
cd ~/Desktop/qwen25_ppo_action_50_000_run/ms-swift
python watch_swift_training_metrics.py --output-dir outputs/qwen25_05b_ppo_action_50_000_rm_a4000
```

### 9. PPO train

RM bittikten sonra:

```bash
MODEL_TYPE=qwen2_5 \
MODEL_PATH=/home/autodidactic/Desktop/qwen25_ep_3500_run/LLaMA-Factory/saves/qwen25_05b_base_full_ft_ep_3500_a4000 \
REWARD_MODEL_PATH=/home/autodidactic/Desktop/qwen25_ep_3500_run/LLaMA-Factory/saves/qwen25_05b_base_full_ft_ep_3500_a4000 \
REWARD_ADAPTERS_PATH=outputs/qwen25_05b_ppo_action_50_000_rm_a4000/last \
./run_ppo_action_50_000_a4000.sh
```

### 10. PPO metriclerini izle

Ayri terminal:

```bash
conda activate qwen25_ppo
cd ~/Desktop/qwen25_ppo_action_50_000_run/ms-swift
python watch_swift_training_metrics.py --output-dir outputs/qwen25_05b_ppo_action_50_000_a4000
```

### 11. Grafik ciz

RM:

```bash
python plot_swift_training_metrics.py --output-dir outputs/qwen25_05b_ppo_action_50_000_rm_a4000
```

PPO:

```bash
python plot_swift_training_metrics.py --output-dir outputs/qwen25_05b_ppo_action_50_000_a4000
```

### 12. PPO LoRA merge et

```bash
MODEL_TYPE=qwen2_5 \
MODEL_PATH=/home/autodidactic/Desktop/qwen25_ep_3500_run/LLaMA-Factory/saves/qwen25_05b_base_full_ft_ep_3500_a4000 \
./export_ppo_action_50_000_lora.sh
```

Merge sonrasi model:

- `exports/qwen25_05b_ppo_action_50_000_a4000_merged`

### 13. Final modeli test et

```bash
python simple_lunarlander_ppo_inference_test.py \
  --model-dir exports/qwen25_05b_ppo_action_50_000_a4000_merged \
  --test-file data/action_50_000_ppo_eval.json \
  --show-examples 5
```

### 14. Zip al

```bash
python archive_ppo_model.py \
  --model-dir exports/qwen25_05b_ppo_action_50_000_a4000_merged
```

## Kisa ozet

Bu klasorde sira su:

1. dataset hazirla
2. RM train
3. PPO train
4. grafiklere bak
5. merge et
6. inference ile test et
7. zip al
