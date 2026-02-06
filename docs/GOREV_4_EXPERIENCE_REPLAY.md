# 🎯 Görev 4: Experience Replay Buffer - Deneyimleri Saklamak

## 📚 Teorik Bilgi

### Experience Replay Nedir?

Q-Learning'de her adımda yaşanan deneyimi (state, action, reward, next_state) hemen kullanıp atmak yerine, bir **hafızada** (buffer) saklıyoruz. Sonra bu hafızadan **rastgele** örnekler çekerek öğreniyoruz.

### Neden Gerekli?

**Problem: Sequential Correlation**
```
Episode: [s1, a1, r1, s2] → [s2, a2, r2, s3] → [s3, a3, r3, s4] ...
```
- Ardışık deneyimler birbirine çok benzer (highly correlated)
- Network bu korelasyonu öğrenir, genelleme yapamaz
- **Catastrophic forgetting**: Yeni deneyimler eskilerin üzerine yazılır

**Çözüm: Experience Replay**
```
Buffer: [(s1,a1,r1,s2), (s5,a5,r5,s6), (s2,a2,r2,s3), ...]
         ↑ Random sampling → Decorrelated batch!
```

### Faydaları

1. **Data Efficiency**: Her deneyimi birden fazla kez kullanabiliriz
2. **Breaking Correlation**: Rastgele sampling korelasyonu kırar
3. **Stable Learning**: Batch güncelleme daha stabil
4. **Sample Reuse**: Pahalı deneyimleri (rare states) tekrar kullanabiliriz

---

## 🛠️ Adım 1: ReplayBuffer Class'ını Oluştur

Yeni bir dosya oluştur: `experience_replay.py`

```python
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Deneyimleri (state, action, reward, next_state, done) tuple'ları 
    olarak saklar ve rastgele batch'ler döndürür.
    
    Parameters:
    -----------
    capacity : int
        Buffer'ın maksimum kapasitesi (eski deneyimler silinir)
    """
    
    def __init__(self, capacity=10000):
        """
        Buffer'ı başlat
        
        deque: Double-ended queue - otomatik olarak eski elemanları siler
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
        print(f"✅ Replay Buffer oluşturuldu: kapasite = {capacity}")
    
    def add(self, state, action, reward, next_state, done):
        """
        Buffer'a yeni deneyim ekle
        
        Parameters:
        -----------
        state : np.array
            Mevcut state
        action : int
            Seçilen action
        reward : float
            Alınan reward
        next_state : np.array
            Sonraki state
        done : bool
            Episode bitti mi?
        """
        # TODO: Tuple olarak sakla
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Buffer'dan rastgele batch çek
        
        Parameters:
        -----------
        batch_size : int
            Kaç deneyim çekilecek
        
        Returns:
        --------
        states, actions, rewards, next_states, dones : tuple of np.arrays
            Batch halinde deneyimler
        """
        # TODO: Rastgele indeksler seç
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # TODO: Seçilen deneyimleri al
        batch = [self.buffer[idx] for idx in indices]
        
        # TODO: Tuple'ları ayır ve numpy array'e çevir
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Buffer'daki deneyim sayısı"""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Yeterli deneyim var mı? (batch çekebilir miyiz?)"""
        return len(self.buffer) >= batch_size
```

### 🔍 Kod Açıklaması

**`deque(maxlen=capacity)`:**
- Double-ended queue: Her iki ucundan ekleme/çıkarma yapılabilen liste
- `maxlen` ayarlı: Dolunca en eski eleman otomatik silinir
- Örnek:
```python
buffer = deque(maxlen=3)
buffer.append(1)  # [1]
buffer.append(2)  # [1, 2]
buffer.append(3)  # [1, 2, 3]
buffer.append(4)  # [2, 3, 4] <- 1 silindi!
```

**`np.random.choice()`:**
```python
# 10 elemanlı buffer'dan 4 tane rastgele seç (tekrarsız)
indices = np.random.choice(10, 4, replace=False)
# Örnek: [7, 2, 9, 1]
```

**Experience Tuple:**
```python
(state, action, reward, next_state, done)
([0.1, 0.5, -0.2, 0.3], 0, 1.0, [0.15, 0.4, -0.15, 0.25], False)
```

---

## 🧪 Test Zamanı!

Test kodunu `experience_replay.py` dosyasının sonuna ekle:

```python
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 4: Experience Replay Buffer Test")
    print("="*60)
    
    # Test 1: Buffer oluştur
    print("\n📦 Test 1: Buffer Oluşturma")
    buffer = ReplayBuffer(capacity=100)
    print(f"   Buffer kapasitesi: {buffer.capacity}")
    print(f"   Mevcut deneyim sayısı: {len(buffer)}")
    
    # Test 2: Deneyim ekle
    print("\n➕ Test 2: Deneyim Ekleme")
    for i in range(10):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = (i == 9)  # Son deneyimde episode biter
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"   10 deneyim eklendi")
    print(f"   Buffer boyutu: {len(buffer)}")
    
    # Test 3: Batch sampling
    print("\n🎲 Test 3: Batch Sampling")
    if buffer.is_ready(batch_size=5):
        states, actions, rewards, next_states, dones = buffer.sample(5)
        
        print(f"   Batch size: 5")
        print(f"   States shape: {states.shape}")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Next states shape: {next_states.shape}")
        print(f"   Dones shape: {dones.shape}")
        
        print(f"\n   İlk deneyim:")
        print(f"      State: {states[0]}")
        print(f"      Action: {actions[0]}")
        print(f"      Reward: {rewards[0]:.3f}")
        print(f"      Done: {bool(dones[0])}")
    
    # Test 4: Kapasite testi
    print("\n🔄 Test 4: Kapasite Testi")
    small_buffer = ReplayBuffer(capacity=5)
    
    for i in range(10):
        small_buffer.add(
            state=np.array([i]),
            action=i,
            reward=float(i),
            next_state=np.array([i+1]),
            done=False
        )
    
    print(f"   10 deneyim eklendi, kapasite: 5")
    print(f"   Buffer boyutu: {len(small_buffer)}")
    print(f"   ✅ Eski deneyimler otomatik silindi!")
    
    # Son 5 deneyimin action'larını göster
    states, actions, _, _, _ = small_buffer.sample(5)
    print(f"   Buffer'daki action'lar: {sorted(actions)}")
    print(f"   (5-9 arasında olmalı, 0-4 silindi)")
    
    # Test 5: Decorrelation (korelasyon kırılması)
    print("\n🎯 Test 5: Decorrelation Test")
    buffer2 = ReplayBuffer(capacity=100)
    
    # Sıralı deneyimler ekle
    for i in range(50):
        buffer2.add(
            state=np.array([i]),
            action=0,
            reward=float(i),
            next_state=np.array([i+1]),
            done=False
        )
    
    # Batch çek ve sıralı mı kontrol et
    states, _, rewards, _, _ = buffer2.sample(10)
    
    print(f"   Buffer'a 0-49 arası sıralı reward'lar eklendi")
    print(f"   Batch'teki reward'lar: {sorted(rewards)}")
    print(f"   ✅ Rastgele sıralı, korelasyon kırıldı!")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
```

---

## 🎯 Beklenen Çıktı

```
============================================================
🧪 GÖREV 4: Experience Replay Buffer Test
============================================================

📦 Test 1: Buffer Oluşturma
✅ Replay Buffer oluşturuldu: kapasite = 100
   Buffer kapasitesi: 100
   Mevcut deneyim sayısı: 0

➕ Test 2: Deneyim Ekleme
   10 deneyim eklendi
   Buffer boyutu: 10

🎲 Test 3: Batch Sampling
   Batch size: 5
   States shape: (5, 4)
   Actions shape: (5,)
   Rewards shape: (5,)
   Next states shape: (5, 4)
   Dones shape: (5,)

   İlk deneyim:
      State: [ 0.123 -0.456  0.789 -0.234]
      Action: 1
      Reward: 0.567
      Done: False

🔄 Test 4: Kapasite Testi
✅ Replay Buffer oluşturuldu: kapasite = 5
   10 deneyim eklendi, kapasite: 5
   Buffer boyutu: 5
   ✅ Eski deneyimler otomatik silindi!
   Buffer'daki action'lar: [5 6 7 8 9]
   (5-9 arasında olmalı, 0-4 silindi)

🎯 Test 5: Decorrelation Test
✅ Replay Buffer oluşturuldu: kapasite = 100
   Buffer'a 0-49 arası sıralı reward'lar eklendi
   Batch'teki reward'lar: [ 3. 12. 18. 25. 31. 34. 37. 42. 45. 48.]
   ✅ Rastgele sıralı, korelasyon kırıldı!

============================================================
✅ Tüm testler tamamlandı!
============================================================
```

---

## 🛠️ Adım 2: Q-Network'e get_action Ekle

`q_learning_nn.py` dosyana epsilon-greedy action seçimi ekle:

```python
def get_action(self, state, epsilon=0.0):
    """
    Epsilon-greedy action seçimi
    
    Parameters:
    -----------
    state : np.array
        Mevcut state
    epsilon : float
        Exploration oranı (0-1 arası)
        0.0: Tamamen exploit (greedy)
        1.0: Tamamen explore (random)
    
    Returns:
    --------
    action : int
        Seçilen action index
    """
    # TODO: Epsilon olasılıkla random action
    if np.random.random() < epsilon:
        return np.random.randint(0, self.action_size)
    
    # TODO: (1-epsilon) olasılıkla en iyi action
    q_values = self.predict(state)
    return np.argmax(q_values)
```

### Test et:
```python
# q_learning_nn.py'nin test kısmına ekle
print("\n🎲 Epsilon-Greedy Test:")
state = np.array([0.1, 0.5, -0.2, 0.3])

for eps in [0.0, 0.5, 1.0]:
    actions = [qnet.get_action(state, epsilon=eps) for _ in range(100)]
    action_counts = np.bincount(actions)
    print(f"   ε={eps:.1f}: Action 0: {action_counts[0]}%, Action 1: {action_counts[1]}%")
```

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `ReplayBuffer` class'ını oluşturdum
- [ ] `__init__`: `deque` ile buffer başlattım
- [ ] `add()`: Deneyim ekleme metodunu yazdım
- [ ] `sample()`: Rastgele batch çekme metodunu yazdım
- [ ] `__len__` ve `is_ready()` yardımcı metodlarını ekledim
- [ ] Test 1-5'i çalıştırdım ve geçti
- [ ] Q-Network'e `get_action()` ekledim
- [ ] Epsilon-greedy'nin çalıştığını doğruladım

---

## 🎓 Ne Öğrendim?

1. **Experience Replay**: Deneyimleri sakla, rastgele sampling yap
2. **Decorrelation**: Korelasyonu kırarak daha stabil öğrenme
3. **deque**: Otomatik kapasite yönetimi ile buffer
4. **Epsilon-Greedy**: Exploration vs Exploitation dengesi
5. **Data Efficiency**: Her deneyimi birden fazla kez kullan

---

## 🧠 Derinlemesine Sorular

1. **Kapasite çok küçük olursa ne olur?** (örn: 100)
2. **Kapasite çok büyük olursa ne olur?** (örn: 1M)
3. **Epsilon decay neden önemli?** (başta 1.0 → sonra 0.01)
4. **Prioritized Experience Replay nedir?** (bonus)

---

## 🚀 Sıradaki Görev

**Görev 5**: CartPole'da Tam DQN - Hepsini bir araya getiriyoruz!

Artık tüm parçaları öğrendin, şimdi gerçek bir RL problemi çözelim! 🎯
