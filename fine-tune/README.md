# Fine-Tune Workspace

Bu klasor, fine-tuning ile ilgili dosyalari daha okunur bir yapida toplar.

## Klasor Yapisi

- `notebooks/`
  - Deney notebook'lari
- `dashboards/`
  - Veri seti dashboard ureten Python script'leri
- `reports/`
  - Uretilen HTML dashboard raporlari
- `training/local/`
  - Yerelde calistirilacak egitim ve test script'leri
- `training/colab/qwen25_05b/`
  - Colab odakli Qwen2.5-0.5B training paketi
- `docs/`
  - README / model card / dokuman taslaklari
- `dataset_samples/`
  - Ornek veri seti dosyalari
- `workspaces/`
  - Yerel, gecici ve ignore edilen notebook calisma alanlari

## En Cok Kullanilan Dosyalar

- Colab full FT notebook:
  - `training/colab/qwen25_05b/Qwen_2_5_0_5B_LunarLander_FullFT_and_Scratch_Colab.ipynb`
- Qwen training gereksinim notlari:
  - `training/colab/qwen25_05b/Qwen_2_5_0_5B_LunarLander_Training_Requirements.txt`
- Dashboard script'leri:
  - `dashboards/`
- HTML raporlar:
  - `reports/`
