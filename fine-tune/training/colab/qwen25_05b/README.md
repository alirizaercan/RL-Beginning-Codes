# Qwen2.5-0.5B LunarLander Training Pack

Bu klasor, `Ali2023kosemen/lunar_lander_270_reward` veri seti icin hazirlanan Colab odakli egitim dosyalarini tek yerde toplar.
Ayri olarak A4000/server uzerinde terminalden calisacak tamamlayici scriptleri de icerir.

Guncel surumde ana akista su duzeltmeler yapildi:
- `Qwen/Qwen2.5-0.5B-Instruct` yerine `Qwen/Qwen2.5-0.5B` base model hedeflenir.
- Veri seti train/eval olarak ayrilir.
- Assistant hedefleri `Action: <id>` formatina kanoniklestirilir.
- Egitim sirasinda `eval loss` loglanir.

## Hangi dosyayi Colab'e yukleyecegim?
Sadece su notebook'u yuklemen yeterli:

- `Qwen_2_5_0_5B_LunarLander_FullFT_Colab.ipynb`

Notebook kendi calismasi sirasinda:
- veri setini `/content/LLaMA-Factory/data/` altina yazar,
- veri setindeki rol adlarini `human/gpt` -> `user/assistant` seklinde normalize eder,
- YAML referans config'ini `/content/LLaMA-Factory/` altina koyar,
- `torchrun src/train.py` komutunu calistiran shell script uretir,
- egitimden sonra loss loglarini okur,
- inference testleri yapar,
- ve istenirse modeli zip'leyip Colab uzerinden indirmeyi dener.

Bu nedenle `.yaml` ya da `.sh` dosyasini ayri ayri Colab'e yuklemen gerekmez.

## Klasordeki dosyalar
- `Qwen_2_5_0_5B_LunarLander_FullFT_Colab.ipynb`
  - Ana notebook. Resmi LLaMA-Factory SFT akisina yakin tek notebook.
- `Qwen_2_5_0_5B_LunarLander_Training_Requirements.txt`
  - GPU gereksinimleri, tahmini sureler ve teknik notlar.
- `A4000_RUNBOOK.md`
  - Shared server / A4000 icin adim adim terminal akisi.
- `qwen25_05b_full_ft_lunarlander_t4.yaml`
  - Notebook'un uretecegi full FT config'in referans kopyasi.
- `run_full_ft_t4_torchrun.sh`
  - `torchrun + src/train.py` yapisina uygun tek GPU Colab T4 komutu.
- `run_full_ft_a4000_torchrun.sh`
  - A4000/server icin terminalden calistirilacak egitim komutu.
- `prepare_lunarlander_dataset_for_llamafactory.py`
  - HF dataset'ini LLaMA-Factory Alpaca formatina, train/eval split ile hazirlar.
- `plot_training_loss.py`
  - Egitim bittikten sonra train ve eval loss grafigi cizer.
- `watch_training_metrics.py`
  - Egitim sirasinda log dosyasini takip eder, son metrikleri yazar ve yeni checkpoint olustukca grafigi gunceller.
- `test_finetuned_lunarlander_model.py`
  - Egitim bittikten sonra coklu prompt ile inference testi yapar.
- `simple_lunarlander_inference_test.py`
  - Base model + alpaca template ile uyumlu daha sade accuracy testi.
- `archive_finetuned_model.py`
  - En son kaydedilen modeli zip'ler.
- `train_qwen25_05b_scratch_lunarlander.py`
  - Ayrica tutulan scratch egitim scripti. Ana notebook bunu kullanmaz.
- `references/llama_factory.rst`
  - Referans dokuman.

## Calistirma sirasi
1. Colab'de GPU olarak `T4` sec.
2. Notebook'u yukle ve ac.
3. GPU kontrol hucresini calistir.
4. Installation hucrelerini calistir.
5. Data Preparation hucrelerini calistir.
6. `torchrun src/train.py` egitim hucresini calistir.
7. Train/eval loss grafigi ve log hucresini calistir.
8. Held-out eval veya inference test hucrelerini calistir.
9. Sonuclar iyiyse modeli zip'leyip indir.

## Neden `torchrun src/train.py`?
Ogretmenin sordugu `torch.cuda` ve `train.py` akisina daha yakin olmak icin notebook egitimi
`torchrun src/train.py` uzerinden baslatir.

## Neden rol adlari donusturuluyor?
HF veri setindeki mesajlar `human` ve `gpt` etiketleriyle geliyor.
Notebook bunlari resmi dokumandaki ShareGPT ornegine daha yakin olmak icin
`user` ve `assistant` olarak yazar.

## Neden bazi argumanlar docs ile birebir ayni degil?
Colab T4 tek GPU senaryosu icin asagidaki iki arguman bilerek eklenmedi:
- `--deepspeed`: tek GPU T4 senaryosunda kurulumu ve hata yuzeyi gereksiz yere artirir.
- `--flash_attn`: Colab T4 tarafinda ek kurulum ve uyumluluk sorunu yaratabildigi icin daha guvenli olan varsayilan attention yolu kullanilir.

Ayrica:
- `--finetuning_type full` kullanilir, cunku hedefimiz LoRA degil full fine-tuning.
- `--fp16` kullanilir, cunku Colab T4 icin daha uygun secim budur.
