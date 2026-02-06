# 🎯 Görev 2: Forward Pass - State'den Q-Value'lara

## 📚 Teorik Bilgi

### Forward Pass Nedir?
**Forward Pass**, neural network'te input'tan output'a giden yolculuktur. Bizim durumumuzda:
- **Input**: State (ortamdan gelen gözlem)
- **Output**: Q-values (her action için beklenen toplam reward)

### Matematiksel İşlemler

#### Layer 1: Input → Hidden
```
z1 = state @ W1 + b1        # Linear transformation
a1 = ReLU(z1) = max(0, z1)  # Activation function
```

#### Layer 2: Hidden → Output
```
q_values = a1 @ W2 + b2     # Linear transformation (no activation)
```

### ReLU Activation Neden?
- **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`
- Negatif değerleri 0 yapar, pozitif değerleri aynen geçirir
- **Neden kullanıyoruz?**
  - Basit ve hızlı hesaplanır
  - Gradient vanishing problemini azaltır
  - Deep learning'de en popüler activation

### Görsel Akış
```
State [0.1, 0.5, -0.2, 0.3]
    ↓ (@ W1 + b1)
Hidden Layer (z1) [..., ..., ...] (32 nöron)
    ↓ (ReLU)
Hidden Layer (a1) [..., ..., ...] (32 nöron, negatifler 0)
    ↓ (@ W2 + b2)
Q-Values [2.3, -1.5]
    ↓
Action: argmax = 0 (sol)
```

---

## 🛠️ Adım 1: Forward Metodunu Yaz

`q_learning_nn.py` dosyasına `forward` metodunu ekle:

```python
def forward(self, state):
    """
    Forward pass - State'den Q-value'lara git
    
    Parameters:
    -----------
    state : np.array
        Shape: (state_size,) veya (batch_size, state_size)
        Örnek: [0.1, 0.5, -0.2, 0.3] - CartPole state'i
    
    Returns:
    --------
    q_values : np.array
        Her action için Q-value tahmini
        Shape: (1, action_size) veya (batch_size, action_size)
    """
    # TODO: State'i 2D yap (batch olarak işle)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    # TODO: Layer 1 - Input → Hidden
    # z1 = state @ W1 + b1
    self.z1 = np.dot(state, self.W1) + self.b1
    
    # TODO: ReLU activation
    # a1 = max(0, z1)
    self.a1 = np.maximum(0, self.z1)
    
    # TODO: Layer 2 - Hidden → Output
    # q_values = a1 @ W2 + b2
    self.q_values = np.dot(self.a1, self.W2) + self.b2
    
    # Son state'i sakla (backward pass için lazım olacak)
    self.last_state = state
    
    return self.q_values
```

### 🔍 Kod Açıklaması

**Neden `state.reshape(1, -1)`?**
```python
# 1D array:
state = [0.1, 0.5, -0.2, 0.3]  # Shape: (4,)

# 2D array (batch):
state = [[0.1, 0.5, -0.2, 0.3]]  # Shape: (1, 4)
```
- Matrix multiplication için 2D gerekli
- `(1, 4) @ (4, 32) = (1, 32)` ✅
- `(4,) @ (4, 32) = Hata!` ❌

**`np.dot` vs `@`**
- İkisi de aynı: matrix multiplication
- `np.dot(A, B)` = `A @ B`

**`np.maximum(0, z1)` - ReLU**
```python
z1 = [-2, 0, 3, -1, 5]
a1 = np.maximum(0, z1)  # [0, 0, 3, 0, 5]
```

---

## 🛠️ Adım 2: Predict Metodunu Ekle

Basit bir wrapper metod - daha okunaklı:

```python
def predict(self, state):
    """
    State için Q-values tahmin et
    
    Parameters:
    -----------
    state : np.array
        State vektörü
    
    Returns:
    --------
    q_values : np.array
        Her action için Q-value
    """
    return self.forward(state)
```

---

## 🧪 Test Zamanı!

Test kodunu dosyanın sonuna ekle:

```python
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 2: Forward Pass Test")
    print("="*60)
    
    # Network oluştur
    qnet = QNetwork(state_size=4, action_size=2, hidden_size=32, learning_rate=0.01)
    
    # Test 1: Tek state
    print("\n📍 Test 1: Tek State")
    state = np.array([0.1, 0.5, -0.2, 0.3])
    print(f"   Input State: {state}")
    print(f"   State Shape: {state.shape}")
    
    q_values = qnet.predict(state)
    print(f"   Output Q-Values: {q_values[0]}")
    print(f"   Q-Values Shape: {q_values.shape}")
    print(f"   En iyi Action: {np.argmax(q_values)}")
    
    # Test 2: Batch processing (birden fazla state)
    print("\n📦 Test 2: Batch Processing")
    batch_states = np.array([
        [0.1, 0.5, -0.2, 0.3],
        [-0.3, 0.2, 0.1, -0.5],
        [0.0, 0.0, 0.0, 0.0]
    ])
    print(f"   Batch Shape: {batch_states.shape}")
    
    batch_q_values = qnet.predict(batch_states)
    print(f"   Batch Q-Values Shape: {batch_q_values.shape}")
    print(f"   Q-Values:")
    for i, q in enumerate(batch_q_values):
        print(f"      State {i}: {q} → Action: {np.argmax(q)}")
    
    # Test 3: ReLU'nun çalıştığını göster
    print("\n🔬 Test 3: ReLU Aktivasyonu")
    print(f"   Hidden Layer (z1) - İlk 5 değer:")
    print(f"      Önce: {qnet.z1[0, :5]}")
    print(f"      Sonra (ReLU): {qnet.a1[0, :5]}")
    print(f"   ✅ Negatif değerler 0 oldu!")
    
    # Test 4: Aynı state her zaman aynı Q-value verir mi?
    print("\n🔁 Test 4: Determinism (Belirlilik)")
    q1 = qnet.predict(state)
    q2 = qnet.predict(state)
    print(f"   İlk tahmin: {q1[0]}")
    print(f"   İkinci tahmin: {q2[0]}")
    print(f"   Aynı mı? {np.allclose(q1, q2)}")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
```

---

## 🎯 Beklenen Çıktı

```
============================================================
🧪 GÖREV 2: Forward Pass Test
============================================================
✅ Ağırlıklar başlatıldı!
   W1 shape: (4, 32) - Input → Hidden
   W2 shape: (32, 2) - Hidden → Output

📍 Test 1: Tek State
   Input State: [ 0.1  0.5 -0.2  0.3]
   State Shape: (4,)
   Output Q-Values: [ 0.23456  -0.12345]
   Q-Values Shape: (1, 2)
   En iyi Action: 0

📦 Test 2: Batch Processing
   Batch Shape: (3, 4)
   Batch Q-Values Shape: (3, 2)
   Q-Values:
      State 0: [ 0.23456  -0.12345] → Action: 0
      State 1: [-0.15678   0.34567] → Action: 1
      State 2: [ 0.01234  -0.05678] → Action: 0

🔬 Test 3: ReLU Aktivasyonu
   Hidden Layer (z1) - İlk 5 değer:
      Önce: [-0.12  0.45 -0.23  0.78  0.11]
      Sonra (ReLU): [0.    0.45  0.    0.78  0.11]
   ✅ Negatif değerler 0 oldu!

🔁 Test 4: Determinism (Belirlilik)
   İlk tahmin: [ 0.23456  -0.12345]
   İkinci tahmin: [ 0.23456  -0.12345]
   Aynı mı? True

============================================================
✅ Tüm testler tamamlandı!
============================================================
```

*(Not: Sayılar random olduğu için seninkiler farklı olacak, ama şekiller aynı olmalı!)*

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `forward()` metodunu ekledim
- [ ] State'i 2D hale getirme işlemini yaptım
- [ ] Layer 1: `z1 = state @ W1 + b1` hesapladım
- [ ] ReLU activation uyguladım: `a1 = max(0, z1)`
- [ ] Layer 2: `q_values = a1 @ W2 + b2` hesapladım
- [ ] `predict()` wrapper metodunu ekledim
- [ ] Test kodunu çalıştırdım
- [ ] Batch processing'in çalıştığını doğruladım
- [ ] ReLU'nun negatif değerleri 0 yaptığını gördüm

---

## 🎓 Ne Öğrendim?

1. **Forward Pass**: Input → Hidden (ReLU) → Output akışı
2. **Matrix Multiplication**: `np.dot()` veya `@` operatörü ile
3. **ReLU Activation**: Negatif değerleri kırparak non-linearity ekler
4. **Batch Processing**: Birden fazla state'i aynı anda işleyebiliriz
5. **Shape Management**: 1D array'i 2D yapmanın önemi

---

## 🧠 Derinlemesine Sorular

Forward pass'i gerçekten anladın mı? Kendine sor:

1. **Neden ReLU kullanıyoruz?** Output layer'da neden yok?
2. **Batch size ne işe yarıyor?** Neden tek tek değil de batch?
3. **`self.z1` ve `self.a1`'i neden saklıyoruz?** Backward pass'te lazım olacak!

---

## 🚀 Sıradaki Görev

**Görev 3**: Backward Pass & Loss - Network nasıl öğreniyor?

Hazır olduğunda bana haber ver! 🎯
