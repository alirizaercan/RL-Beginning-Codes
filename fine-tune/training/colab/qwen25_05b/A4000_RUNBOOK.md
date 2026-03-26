# Qwen2.5-0.5B Full FT on A4000

Bu not, `Ali2023kosemen/lunar_lander_270_reward` veri setiyle
`Qwen/Qwen2.5-0.5B` base modelini **LLaMA-Factory full SFT**
olarak A4000 uzerinde calistirmak icin hazirlandi.

Bu ikinci surumde uc kritik duzeltme var:
- `Instruct` yerine `Base` model kullaniliyor.
- Veri seti `train/eval` olarak sabit bicimde ayriliyor.
- Assistant cevabi serbest metin yerine kanonik olarak `Action: <id>` bicimine indiriliyor.

## Hangi dosyayi gonderecegim?
Server'a notebook gondermek zorunda degilsin.

Referans icin:
- `Qwen_2_5_0_5B_LunarLander_FullFT_Colab.ipynb`

Server'da gercekten kullanilacak dosyalar:
- `prepare_lunarlander_dataset_for_llamafactory.py`
- `run_full_ft_a4000_torchrun.sh`
- `plot_training_loss.py`
- `watch_training_metrics.py`
- `simple_lunarlander_inference_test.py`
- `archive_finetuned_model.py`

## Neden notebook yerine script?
- Notebook Colab'e ozel `/content/...` yollarini kullaniyor.
- Shared sirket makinesinde terminal + conda env daha guvenli.
- Diger calisanlarin ortamini bozmadan kendi izole env'inde calisirsin.

## 1. Temiz env ac
```bash
conda create -n qwen25_fullft python=3.11 -y
conda activate qwen25_fullft
python --version
```

## 2. Kendi calisma klasorune gec
Ornek:
```bash
mkdir -p ~/qwen25_fullft_run
cd ~/qwen25_fullft_run
```

## 3. LLaMA-Factory kur
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -U pip
pip install -e ".[torch,metrics]"
pip install datasets huggingface_hub pandas matplotlib accelerate
```

## 4. CUDA test et
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))"
```

Beklenen:
- `True`
- GPU adi: `NVIDIA RTX A4000`

## 5. Gerekli dosyalari kopyala
Bu repodan su dosyalari server'daki `LLaMA-Factory` klasorune kopyala:
- `fine-tune/training/colab/qwen25_05b/prepare_lunarlander_dataset_for_llamafactory.py`
- `fine-tune/training/colab/qwen25_05b/run_full_ft_a4000_torchrun.sh`
- `fine-tune/training/colab/qwen25_05b/plot_training_loss.py`
- `fine-tune/training/colab/qwen25_05b/watch_training_metrics.py`
- `fine-tune/training/colab/qwen25_05b/simple_lunarlander_inference_test.py`
- `fine-tune/training/colab/qwen25_05b/archive_finetuned_model.py`

Son durumda `LLaMA-Factory` klasoru icinde bunlar olmali:
- `src/train.py` zaten repo ile birlikte gelir, ayri kopyalaman gerekmez
- `prepare_lunarlander_dataset_for_llamafactory.py`
- `run_full_ft_a4000_torchrun.sh`
- `plot_training_loss.py`
- `watch_training_metrics.py`
- `simple_lunarlander_inference_test.py`
- `archive_finetuned_model.py`

## 6. Dataset'i hazirla
`LLaMA-Factory` klasoru icinden:
```bash
python prepare_lunarlander_dataset_for_llamafactory.py
```

Bu komut:
- `data/lunar_lander_270_reward_train.json`
- `data/lunar_lander_270_reward_eval.json`
- `data/lunar_lander_270_reward_full.json`
- `data/dataset_info.json`
dosyalarini hazirlar.

Not:
- Train/eval split deterministiktir.
- Cevaplar `Action: 0`, `Action: 1`, `Action: 2`, `Action: 3` formatina indirgenir.
- Bu sayede training loss ile action accuracy birbirine oncekine gore daha iyi hizalanir.

## 7. Shared makinede dikkat edilmesi gerekenler
- `base` env icinde calisma.
- Sisteme global `pip install` yapma.
- Baska bir ekibin GPU isini kapatma.
- Eger Firefox / VS Code / baska Python processleri sana aitse ve gereksizse kapat.
- Egitimi tercihen `tmux` icinde baslat.

## 8. Egitimi baslat
```bash
chmod +x run_full_ft_a4000_torchrun.sh
./run_full_ft_a4000_torchrun.sh
```

## 9. Izleme
Ayri terminalde:
```bash
watch -n 2 nvidia-smi
```

veya:
```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 2
```

Egitim metriklerini ve checkpoint bazli guncellenen grafigi izlemek icin:
```bash
python watch_training_metrics.py --output-dir saves/qwen25_05b_base_full_ft_lunarlander_a4000
```

Bu script:
- `trainer_log.jsonl` dosyasini takip eder
- son `train loss`, `eval loss`, `eval accuracy`, `grad norm`, `learning rate` degerlerini yazdirir
- yeni checkpoint olusunca `training_loss.png` dosyasini yeniden cizer

## 10. Egitimden sonra loss grafigi cikar
```bash
python plot_training_loss.py
```

Bu komut:
- `trainer_log.jsonl` dosyasini okur
- train ve eval loss grafigini cizer
- varsa eval accuracy cizgisini de ekler
- `training_loss.png` olarak kaydeder

## 11. Egitimden sonra inference testi yap
Held-out eval split ile basit accuracy kontrolu yapmak icin:
```bash
python simple_lunarlander_inference_test.py \
  --model-dir saves/qwen25_05b_base_full_ft_lunarlander_a4000/checkpoint-6250 \
  --test-file data/lunar_lander_270_reward_eval.json
```

Not:
- Bu script `alpaca` template ile uyumlu prompt kurar.
- `test-file` olarak dogrudan eval json verilebilir.
- Her satirda beklenen action ile tahmin edilen action karsilastirilir.

## 12. Egitimden sonra modeli arsivle
```bash
python archive_finetuned_model.py
```

Bu komut en son checkpoint/model klasorunu zip'ler.

## 13. Sonra requirements cikar
Her sey calisiyorsa:
```bash
pip freeze > requirements.lock.txt
conda env export --no-builds > environment.yml
```

## Beklenti
- T4'te calisan ayarlarla A4000'de de yuksek olasilikla calisir.
- A4000 16 GB oldugu icin mevcut ayarlar konservatif ve guvenlidir.
- Buyuk fark bekleme, ama T4'ten bir miktar daha iyi performans gorebilirsin.
- Onceki surume gore daha anlamli bir karsilastirma icin artik `eval loss` ve held-out accuracy birlikte gorulecek.
