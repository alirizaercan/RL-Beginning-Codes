# 🎯 Görev 1: Q-Learning Neural Network - Temel Mimari

## 📚 Teorik Bilgi

### Q-Learning Nedir?
Q-Learning bir **değer tabanlı** RL algoritmasıdır. Her (state, action) çifti için bir "Q-value" öğrenir:
- **Q(s,a)**: State `s`'de action `a`'yı seçmenin ne kadar iyi olduğunu gösterir
- Amaç: Her durumda en yüksek toplam reward'ı verecek action'ları öğrenmek

### Q-Network Neden Gerekli?
- **Klasik Q-Learning**: Tablo kullanır (her state-action için bir değer)
- **Sorun**: Büyük state space'lerde (örn: görüntüler) tablo çok büyük olur
- **Çözüm**: Neural Network ile Q-value'ları **tahmin** et!

### Q-Network Mimarisi
```
State (Input)  →  Hidden Layer(s)  →  Q-Values (Output)
  [4 dim]            [64 neurons]         [2 dim]
                      (ReLU)             (Linear)
```

**Örnek (CartPole):**
- Input: `[position, velocity, angle, angular_velocity]` (4 değer)
- Output: `[Q(s, left), Q(s, right)]` (2 Q-value)

---

## 🛠️ Adım 1: Class Yapısını Oluştur

**Dosya:** `q_learning_nn.py`

```python
import numpy as np

class QNetwork:
    """
    Q-Learning için Neural Network
    
    Parametreler:
    - state_size: State vektörünün boyutu (örn: CartPole'da 4)
    - action_size: Kaç farklı action var (örn: CartPole'da 2)
    - hidden_size: Hidden layer'da kaç nöron (default: 64)
    - learning_rate: Öğrenme hızı (default: 0.001)
    """
    
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001):
        # TODO: Parametreleri sakla
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # Ağırlıkları başlat (sonraki adımda yapacağız)
        pass
```

### 🧪 Test Et
```python
# Dosyanın sonuna ekle
if __name__ == "__main__":
    qnet = QNetwork(state_size=4, action_size=2, hidden_size=32)
    print(f"✅ Q-Network oluşturuldu: {qnet.state_size} → {qnet.hidden_size} → {qnet.action_size}")
```

**Çalıştır:**
```bash
python neural-network/q_learning_nn.py
```

**Beklenen Çıktı:**
```
✅ Q-Network oluşturuldu: 4 → 32 → 2
```

---

## 🛠️ Adım 2: Ağırlıkları (Weights) Başlat

Neural network'ün "beynini" oluşturuyoruz! İki katman var:
1. **Input → Hidden**: `W1` ve `b1`
2. **Hidden → Output**: `W2` ve `b2`

### Neden Random Başlatıyoruz?
- Eğer tüm ağırlıklar 0 olsaydı, tüm nöronlar aynı şeyi öğrenirdi
- Random başlatmak **simetri kırılmasını** sağlar

### He Initialization
- ReLU activation için özel bir başlatma yöntemi
- Formula: `std = sqrt(2 / n_input)`

```python
def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001):
    self.state_size = state_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.lr = learning_rate
    
    # TODO: Ağırlıkları He initialization ile başlat
    
    # Layer 1: Input → Hidden
    self.W1 = np.random.randn(state_size, hidden_size) * np.sqrt(2.0 / state_size)
    self.b1 = np.zeros((1, hidden_size))
    
    # Layer 2: Hidden → Output
    self.W2 = np.random.randn(hidden_size, action_size) * np.sqrt(2.0 / hidden_size)
    self.b2 = np.zeros((1, action_size))
    
    print(f"✅ Ağırlıklar başlatıldı!")
    print(f"   W1 shape: {self.W1.shape} - Input → Hidden")
    print(f"   W2 shape: {self.W2.shape} - Hidden → Output")
```

### 📊 Shape'leri Anlamak
```
State (4,)  →  W1 (4, 32)  →  Hidden (32,)  →  W2 (32, 2)  →  Q-values (2,)
```

### 🧪 Test Et
```python
if __name__ == "__main__":
    qnet = QNetwork(state_size=4, action_size=2, hidden_size=32)
    
    # Ağırlık şekillerini kontrol et
    print(f"\n🔍 W1 shape: {qnet.W1.shape}")  # Beklenen: (4, 32)
    print(f"🔍 W2 shape: {qnet.W2.shape}")    # Beklenen: (32, 2)
    print(f"🔍 b1 shape: {qnet.b1.shape}")    # Beklenen: (1, 32)
    print(f"🔍 b2 shape: {qnet.b2.shape}")    # Beklenen: (1, 2)
```

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `QNetwork` class'ını oluşturdum
- [ ] `__init__` metodunda parametreleri sakladım
- [ ] `W1`, `b1`, `W2`, `b2` ağırlıklarını He initialization ile başlattım
- [ ] Test kodunu çalıştırdım ve shape'leri doğruladım
- [ ] Ağırlıkların random değerler içerdiğini kontrol ettim

---

## 🎓 Ne Öğrendim?

1. **Q-Network'ün amacı**: State alıp, her action için Q-value üretmek
2. **Mimari**: 2 katmanlı basit bir NN (Input → Hidden → Output)
3. **He Initialization**: ReLU ile kullanılan özel ağırlık başlatma
4. **Shape'ler**: Matrix multiplication için boyutların nasıl uyması gerektiği

---

## 🚀 Sıradaki Görev

**Görev 2**: Forward Pass - State'den Q-value'lara nasıl gidilir?

Hazır olduğunda bana haber ver! 🎯
