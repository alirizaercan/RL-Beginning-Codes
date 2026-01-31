# RL-Beginning-Codes

Bu repo, Reinforcement Learning (Takviyeli Öğrenme) algoritmalarının temel implementasyonlarını içermektedir.

## 📁 Proje Yapısı

```
RL-Beginning-Codes/
├── frozen-lake/           # Frozen Lake ortamı için RL algoritmaları
│   ├── frozen_lake_q.py       # Q-Learning implementasyonu
│   ├── frozen_lake_dql.py     # Deep Q-Learning implementasyonu
│   └── frozen_lake_dql.pt     # Eğitilmiş DQL model
│
├── lunar-lander/          # Lunar Lander ortamı için RL algoritmaları
│   ├── lunar_lander_ppo_gym.py         # PPO algoritması (Gym ortamı)
│   ├── lunar_lander_ppo_transformer.py # PPO Transformer varyasyonu
│   └── validation.py                    # Ortam validasyon scripti
│
├── docs/                  # Dökümanlar ve notlar
│   ├── PPO_ANALYSIS_TR.md       # PPO algoritması analiz notları
│   └── PPO_SUNUM_NOTLARI.md     # PPO sunum notları
│
└── outputs/               # Eğitim çıktıları ve metrikler
    └── ppo_outputs/       # PPO eğitim sonuçları
```

## 🎮 Projeler

### 1. Frozen Lake
OpenAI Gymnasium'daki Frozen Lake ortamı üzerinde iki farklı yaklaşım:
- **Q-Learning**: Klasik tablo tabanlı Q-Learning algoritması
- **Deep Q-Learning (DQN)**: Derin sinir ağı kullanarak Q değerlerini öğrenen algoritma

### 2. Lunar Lander
Lunar Lander ortamı için Proximal Policy Optimization (PPO) algoritması implementasyonları:
- **PPO (Gym)**: Standart PPO implementasyonu
- **PPO Transformer**: Transformer mimarisi kullanan PPO varyasyonu

## 🚀 Kullanım

### Gereksinimler
```bash
pip install gymnasium torch numpy matplotlib
```

### Frozen Lake Q-Learning
```bash
cd frozen-lake
python frozen_lake_q.py
```

### Frozen Lake DQN
```bash
cd frozen-lake
python frozen_lake_dql.py
```

### Lunar Lander PPO
```bash
cd lunar-lander
python lunar_lander_ppo_gym.py
```

## 📊 Çıktılar
Eğitim sürecinde oluşan metrikler ve grafikler `outputs/` klasöründe saklanır.

## 📖 Dökümanlar
Algoritmaların detaylı analizleri ve notları için `docs/` klasörüne bakabilirsiniz.

## 📝 Lisans
Bu proje eğitim amaçlıdır.
