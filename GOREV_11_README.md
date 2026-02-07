# ✅ Frame Stacking CartPole Lab - Hazır!

## 📄 Oluşturulan Dosya

**`docs/GOREV_11_FRAME_STACKING_CARTPOLE.md`** (987 satır)

Bu tek markdown dosyası CartPole oyunu için frame stacking PPO implementasyonunun tüm teoriyi ve pratiğini içeriyor.

---

## 📋 Dosya İçeriği

### 1. **Teori Bölümü**
- 🎯 Görevin amacı
- 📚 Frame stacking nedir? (basit açıklama)
- 🔧 Matematiksel tanım (Deque yöntemi)

### 2. **Implementasyon Rehberi**
- ✅ `FrameStackWrapper` sınıfı (detaylı pseudocode)
  - Tüm metodları açıklanmış
  - Preprocessing mantığı
  - Deque kullanımı
  
- 🧠 PPO Agent Mikarisi
  - Actor network detayı
  - Critic network detayı
  - Visual network diagram'ları

- 🔄 Training Loop
  - Step-by-step eğitim süreci
  - Episode yapısı
  - Weight update'ler

### 3. **CartPole Spesifik Bilgiler**
- 📊 Environment info tablosu
- 🎯 Bounds ve başarı kriterleri
- ⚖️ Normal vs Frame Stacking karşılaştırması
- 📈 Performance metrikleri tablosu

### 4. **Implementasyon Kılavuzu**
- 🛠️ Adım adım kodlama rehberi
- ✅ Imports
- ✅ FrameStackWrapper sınıfı (tam pseudocode)
- ✅ FrameStackPPOAgent sınıfı (tam pseudocode)
- ✅ Main training loop

### 5. **Debugging & Optimization**
- 🔍 3 yaygın problem + çözüm
- ⚡ Hızlandırma teknikleri
- 🧪 Test senaryoları

### 6. **Sonuçlar & Analiz**
- 📊 Expected output örneği
- 📈 Grafik incelemesi
- ❓ Analiz soruları
- 📝 Implementation checklist

### 7. **Referanslar**
- 🔗 İlişkili görevler
- 🎓 Öğrenme çıktıları
- 📞 Debugging rehberi

---

## 🚀 Nasıl Kullanılacak?

### Adım 1: Teoriyi Oku

```bash
# Markdown dosyasını aç
cat docs/GOREV_11_FRAME_STACKING_CARTPOLE.md

# veya preferred editor'de
code docs/GOREV_11_FRAME_STACKING_CARTPOLE.md
```

### Adım 2: Konseptleri Anla

- Frame stacking nedir? (ilk 50 satır)
- Neden Deque kullanılır? (satır ~100)
- PPO mimarisi (satır ~250)

### Adım 3: Implementasyon Yapı

Markdown dosyasındaki "Implementasyon Kılavuzu" bölümünü takip et:
1. `FrameStackWrapper` yaz
2. `FrameStackPPOAgent` yaz
3. Training loop yaz

### Adım 4: Kodunuzu Çalıştır

```python
# your_cartpole_frame_stacking.py

import numpy as np
import gymnasium as gym

# Adım 1-3'deki sınıfları tanımla

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = FrameStackWrapper(env, num_frames=4)
    
    agent = FrameStackPPOAgent(state_size=4*84*84)
    rewards = agent.train(env, episodes=500)
    
    # Sonuçları görselleştir
    plt.plot(rewards)
    plt.show()
```

### Adım 5: Sonuçları Analiz Et

Markdown dosyasının "Sonuçları Analiz Etme" bölümü:
- Learning curve'u incele
- Performance metrikleri kontrol et
- Hiper-parametreleri optimize et

---

## 📊 İçerik Dağılımı

| Bölüm | Satır | Konu |
|-------|-------|------|
| Giriş & Teori | 1-150 | Frame stacking konsepti |
| Implementasyon Teorisi | 150-350 | FrameStackWrapper detayları |
| PPO Architecture | 350-500 | Actor-Critic networks |
| Training Loop | 500-600 | Eğitim süreci |
| CartPole Info | 600-700 | Ortam özellikleri |
| Comparison | 700-800 | Normal vs Frame Stacking |
| Kılavuz & Kod | 800-900 | Step-by-step implementasyon |
| Debugging & Test | 900-950 | Sorun çözme |
| Referanslar | 950-987 | Son öneriler |

---

## 💡 Markdown Dosyasında Ne Var?

### ✅ Pseudocode (Çalışmaya hazır yapı)

```python
# Markdown'da bu yapıda veriliyor:

class FrameStackWrapper:
    def __init__(self, ...):
        # [PSEUDOCODE]
    
    def reset(self):
        # [PSEUDOCODE]
    
    def step(self, action):
        # [PSEUDOCODE]
```

Siz bunu kopyalayıp kendi Python dosyasında çalışan koda çeviriyorsunuz.

### ✅ Visual Diagrams

```
Network Architecture:
┌──────────┐
│ Input    │
└─────┬────┘
      ↓
 ┌─────────┐
 │ Layer 1 │
 └────┬────┘
      ↓
 ┌─────────┐
 │ Layer 2 │
 └────┬────┘
      ↓
 ┌────────────┐
 │ Output     │
 └────────────┘
```

### ✅ Detaylı Açıklamalar

Her bölümde:
- Konsept açıklandı
- Neden böyle? sorusunun cevabı
- Matematiksel formüller
- Kod örnekleri

### ✅ Kontrol Listesi

```
Implementation Checklist:
- [ ] __init__ deque'i initialize ediyor
- [ ] reset() frame'leri dolduruyor
- [ ] step() yeni frame'i ekliyor
- ...
```

---

## 🎯 Başarı Kriterleri

Sonunda başarılı olmuş sayılmak için:

1. ✅ Markdown dosyasını tamamıyla okudun
2. ✅ `FrameStackWrapper` sınıfını yazdın ve çalıştırdın
3. ✅ `FrameStackPPOAgent` sınıfını yazdın ve çalıştırdın
4. ✅ Agent eğitildi (reward artıyor)
5. ✅ Sonuçları görselleştirdin
6. ✅ Normal PPO ile karşılaştırdın

---

## 🔗 İlişkili Dosyalar

- `README.md`: Güncellendi (yeni görev eklendi)
- `neural-network/`: İmplementasyon burada
- `outputs/`: Sonuçlar burada kaydedilecek

---

## 📝 Örnek Başlangıç Kodu

Markdown dosyasını okuduktan sonra, başlayabileceğiniz skeleton:

```python
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# 1. FrameStackWrapper sınıfını yaz
class FrameStackWrapper:
    def __init__(self, env, num_frames=4, frame_size=84):
        # TODO: Markdown'dan kopyala
        pass
    
    def reset(self):
        # TODO: Markdown'dan kopyala
        pass
    
    def step(self, action):
        # TODO: Markdown'dan kopyala
        pass

# 2. FrameStackPPOAgent sınıfını yaz
class FrameStackPPOAgent:
    def __init__(self, state_size=28224, action_size=2):
        # TODO: Markdown'dan kopyala
        pass
    
    def get_action(self, state):
        # TODO: Markdown'dan kopyala
        pass
    
    def train(self, env, episodes=500):
        # TODO: Markdown'dan kopyala
        pass

# 3. Main script
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = FrameStackWrapper(env, num_frames=4)
    
    agent = FrameStackPPOAgent()
    rewards = agent.train(env, episodes=500)
    
    plt.plot(rewards)
    plt.savefig("results.png")
```

---

## 🎓 Ne Öğreneceksiniz?

Markdown dosyasını takip ederek:

✅ Frame stacking nedir ve nasıl çalışır
✅ Deque veri yapısını kullanmak
✅ State'i visual'leştirmek
✅ PPO agent'ini implement etmek
✅ Training loop yazıp optimize etmek
✅ Sonuçları analiz etmek
✅ Debugging & optimization teknikleri

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Dosyayı oku
cat docs/GOREV_11_FRAME_STACKING_CARTPOLE.md

# 2. Kodu yaz
cd neural-network
nano cartpole_frame_stacking.py

# 3. Çalıştır
python cartpole_frame_stacking.py
```

---

**Başlamaya Hazır mısınız? 🚀**

Markdown dosyasının ilk 50 satırını oku ve temel konseptleri anla!
