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
├── neural-network/        # Neural Network tabanlı RL algoritmaları
│   ├── simple_nn.py                  # Basit neural network
│   ├── simple_dqn.py                 # Basit DQN
│   ├── q_learning_nn.py              # Q-Learning + NN
│   ├── experience_replay.py           # Experience Replay mekanizması
│   ├── cartpole_dqn.py               # CartPole ortamı DQN
│   ├── cartpole_ppo.py               # CartPole ortamı PPO
│   ├── ppo_networks.py               # PPO Actor-Critic networks
│   └── ppo_agent.py                  # PPO Agent implementasyonu
│
├── docs/                  # Dökümanlar ve notlar
│   ├── GOREV_1_Q_NETWORK_TEMEL.md
│   ├── GOREV_2_FORWARD_PASS.md
│   ├── GOREV_3_BACKWARD_PASS.md
│   ├── GOREV_4_EXPERIENCE_REPLAY.md
│   ├── GOREV_5_CARTPOLE_DQN.md
│   ├── GOREV_6_PPO_ACTOR_CRITIC_MIMARI.md
│   ├── GOREV_7_PPO_POLICY_GRADIENT.md
│   ├── GOREV_8_PPO_CLIP.md
│   ├── GOREV_9_PPO_TRAINING_LOOP.md
│   ├── GOREV_10_CARTPOLE_PPO.md
│   ├── GOREV_11_FRAME_STACKING_CARTPOLE.md  # Frame Stacking Lab 🆕
│   ├── PPO_ANALYSIS_TR.md
│   └── PPO_SUNUM_NOTLARI.md
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

### 3. CartPole PPO
CartPole oyunu için neural network tabanlı RL algoritmaları:
- **Simple DQN**: DQN'nin basit implementasyonu
- **CartPole PPO**: Tam PPO implementasyonu
- **Frame Stacking PPO**: Video oyunları için frame stacking ile PPO 🆕

## 📚 Görevler (Tasks)

| Görev | Konu | Dosya |
|-------|------|-------|
| Görev 1 | Q-Network Temelleri | `GOREV_1_Q_NETWORK_TEMEL.md` |
| Görev 2 | Forward Pass | `GOREV_2_FORWARD_PASS.md` |
| Görev 3 | Backward Pass | `GOREV_3_BACKWARD_PASS.md` |
| Görev 4 | Experience Replay | `GOREV_4_EXPERIENCE_REPLAY.md` |
| Görev 5 | CartPole DQN | `GOREV_5_CARTPOLE_DQN.md` |
| Görev 6 | PPO Actor-Critic | `GOREV_6_PPO_ACTOR_CRITIC_MIMARI.md` |
| Görev 7 | Policy Gradient | `GOREV_7_PPO_POLICY_GRADIENT.md` |
| Görev 8 | PPO Clipping | `GOREV_8_PPO_CLIP.md` |
| Görev 9 | PPO Training Loop | `GOREV_9_PPO_TRAINING_LOOP.md` |
| Görev 10 | CartPole PPO | `GOREV_10_CARTPOLE_PPO.md` |
| Görev 11 | Frame Stacking CartPole | `GOREV_11_FRAME_STACKING_CARTPOLE.md` 🆕 |

## 🚀 Kullanım

### Gereksinimler
```bash
pip install gymnasium torch numpy matplotlib
```

### Frame Stacking CartPole Lab 🆕

Yeni **Görev 11: Frame Stacking ile CartPole PPO** lab'ını başlat:

```bash
# 1. Lab dökümanını oku
cat docs/GOREV_11_FRAME_STACKING_CARTPOLE.md

# 2. Kendi implementasyonunu yaz (.py dosyası)
# - FrameStackWrapper sınıfı
# - FrameStackPPOAgent sınıfı
# - Training loop

# 3. Çalıştır ve sonuçları gözlemle
python your_cartpole_frame_stacking.py
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

### CartPole PPO
```bash
cd neural-network
python cartpole_ppo.py
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
