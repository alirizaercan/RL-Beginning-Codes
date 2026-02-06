# 🎯 Görev 3: Backward Pass & Loss - Network Nasıl Öğreniyor?

## 📚 Teorik Bilgi

### Öğrenme Süreci

Forward pass yaptık, Q-value tahminleri aldık. Ama network nasıl öğrenecek? **Backward Pass** (Backpropagation)!

### Q-Learning'de Loss Fonksiyonu

**Bellman Equation** (RL'nin kalbi):
```
Q(s, a) = reward + γ × max Q(s', a')
          ↑                    ↑
      Anlık reward      Gelecek en iyi Q-value
```

**Loss (Hata):**
```
Loss = (predicted_q - target_q)²
```

Amaç: Tahmin ettiğimiz Q-value'yu, Bellman equation'dan gelen target'a yaklaştırmak!

### Gradient Descent

Network'ü güncellemek için ağırlıkları küçük adımlarla değiştiriyoruz:

```
W_new = W_old - learning_rate × gradient
```

**Gradient**: Loss fonksiyonunun ağırlıklara göre türevi (hangi yöne gitmeliyiz?)

### Backpropagation Akışı

```
Loss (MSE)
    ↓ (gradient)
Output Layer (W2, b2)
    ↓ (gradient)
Hidden Layer (W1, b1)
    ↓
Weights Updated! ✅
```

---

## 🧮 Matematiksel Detaylar

### Loss: Mean Squared Error (MSE)

```python
Loss = (1/batch_size) × Σ(predicted - target)²
```

**Gradient (türev):**
```python
dLoss/dQ = 2 × (predicted - target) / batch_size
```

### Chain Rule (Zincir Kuralı)

Backpropagation'un temelinde chain rule var:

```
dLoss/dW2 = dLoss/dQ × dQ/dW2
```

### ReLU'nun Türevi

```python
ReLU(x) = max(0, x)

ReLU'(x) = 1 if x > 0
           0 if x ≤ 0
```

Kod olarak:
```python
d_relu = (x > 0).astype(float)  # True→1, False→0
```

---

## 🛠️ Adım 1: Backward Metodunu Yaz

`q_learning_nn.py` dosyasına `backward()` metodunu ekle:

```python
def backward(self, target_q_values):
    """
    Backward pass - Gradient hesapla ve ağırlıkları güncelle
    
    Parameters:
    -----------
    target_q_values : np.array
        Hedef Q-values (Bellman equation'dan gelir)
        Shape: (batch_size, action_size)
    
    Notlar:
    -------
    - Forward pass'ten sonra çağrılmalı (self.last_state, self.z1, self.a1 kullanır)
    - Gradientleri hesaplar ve ağırlıkları günceller (gradient descent)
    """
    batch_size = self.last_state.shape[0]
    
    # TODO: Loss gradient - MSE'nin türevi
    # dL/dQ = 2 * (predicted - target) / batch_size
    dq = 2 * (self.q_values - target_q_values) / batch_size
    
    # TODO: Output layer gradients (W2, b2)
    # dL/dW2 = a1.T @ dq
    # dL/db2 = sum(dq)
    dW2 = np.dot(self.a1.T, dq)
    db2 = np.sum(dq, axis=0, keepdims=True)
    
    # TODO: Hidden layer gradients
    # Önce: dL/da1 = dq @ W2.T
    da1 = np.dot(dq, self.W2.T)
    
    # ReLU backprop: sadece pozitif olan yerlerde gradient geçer
    # dL/dz1 = dL/da1 * (z1 > 0)
    dz1 = da1 * (self.z1 > 0)
    
    # TODO: Input layer gradients (W1, b1)
    # dL/dW1 = state.T @ dz1
    # dL/db1 = sum(dz1)
    dW1 = np.dot(self.last_state.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # TODO: Gradient descent - Ağırlıkları güncelle
    self.W1 -= self.lr * dW1
    self.b1 -= self.lr * db1
    self.W2 -= self.lr * dW2
    self.b2 -= self.lr * db2
```

### 🔍 Kod Açıklaması

**1. Loss Gradient:**
```python
dq = 2 * (self.q_values - target_q_values) / batch_size
```
- MSE'nin türevi: `d/dx[(x-y)²] = 2(x-y)`
- Batch size'a böleriz (ortalama)

**2. Output Layer:**
```python
dW2 = np.dot(self.a1.T, dq)  # (32, 1) @ (1, 2) = (32, 2)
db2 = np.sum(dq, axis=0)     # (1, 2) → (1, 2)
```

**3. ReLU Backprop:**
```python
da1 = np.dot(dq, self.W2.T)   # Gradient'i geriye gönder
dz1 = da1 * (self.z1 > 0)     # ReLU mask: sadece pozitif yerlerde
```

**4. Gradient Descent:**
```python
self.W1 -= self.lr * dW1  # W = W - α × ∇W
```

---

## 🛠️ Adım 2: Öğrenme Testi

Test kodunu ekle - Network'ün gerçekten öğrenip öğrenmediğini görelim!

```python
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 3: Backward Pass & Öğrenme Testi")
    print("="*60)
    
    # Network oluştur
    qnet = QNetwork(state_size=4, action_size=2, hidden_size=32, learning_rate=0.1)
    
    # Test state
    state = np.array([0.1, 0.5, -0.2, 0.3])
    
    # Test 1: Öğrenmeden önce
    print("\n📊 Test 1: Öğrenmeden Önce")
    q_before = qnet.predict(state)
    print(f"   State: {state}")
    print(f"   Q-Values (önce): {q_before[0]}")
    print(f"   Seçilen Action: {np.argmax(q_before)}")
    
    # Test 2: Öğrenme - Action 0'ı tercih etmesini öğretelim
    print("\n🎓 Test 2: Öğrenme - Action 0'ı Ödüllendirme")
    target_q = q_before.copy()
    target_q[0, 0] = 10.0   # Action 0 için yüksek target
    target_q[0, 1] = -5.0   # Action 1 için düşük target
    print(f"   Target Q-Values: {target_q[0]}")
    
    # Backward pass
    qnet.backward(target_q)
    
    # Öğrendikten sonra
    q_after = qnet.predict(state)
    print(f"   Q-Values (sonra): {q_after[0]}")
    print(f"   Seçilen Action: {np.argmax(q_after)}")
    print(f"   ✅ Q-value değişimi:")
    print(f"      Action 0: {q_before[0,0]:.3f} → {q_after[0,0]:.3f} (artmalı)")
    print(f"      Action 1: {q_before[0,1]:.3f} → {q_after[0,1]:.3f} (azalmalı)")
    
    # Test 3: Çoklu iterasyon - Daha iyi öğreniyor mu?
    print("\n🔁 Test 3: 100 İterasyon Öğrenme")
    qnet2 = QNetwork(state_size=4, action_size=2, hidden_size=32, learning_rate=0.01)
    
    initial_q = qnet2.predict(state)
    print(f"   Başlangıç Q-Values: {initial_q[0]}")
    
    target = initial_q.copy()
    target[0, 1] = 20.0  # Action 1'i öğret
    
    losses = []
    for i in range(100):
        # Forward pass
        predicted = qnet2.predict(state)
        
        # Loss hesapla (MSE)
        loss = np.mean((predicted - target)**2)
        losses.append(loss)
        
        # Backward pass
        qnet2.backward(target)
        
        if (i+1) % 20 == 0:
            print(f"   İterasyon {i+1:3d}: Loss = {loss:.6f}, Q = {predicted[0]}")
    
    final_q = qnet2.predict(state)
    print(f"   Final Q-Values: {final_q[0]}")
    print(f"   ✅ Action 1 Q-value: {initial_q[0,1]:.3f} → {final_q[0,1]:.3f}")
    
    # Test 4: Loss grafiği (opsiyonel - matplotlib varsa)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel('İterasyon')
        plt.ylabel('Loss (MSE)')
        plt.title('Öğrenme Eğrisi - Loss Zamanla Azalıyor mu?')
        plt.grid(True)
        plt.savefig('neural-network/learning_curve_q3.png')
        print(f"\n📈 Loss grafiği kaydedildi: neural-network/learning_curve_q3.png")
    except ImportError:
        print("\n⚠️ Matplotlib yok, grafik atlandı")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
```

---

## 🎯 Beklenen Çıktı

```
============================================================
🧪 GÖREV 3: Backward Pass & Öğrenme Testi
============================================================
✅ Ağırlıklar başlatıldı!
   W1 shape: (4, 32) - Input → Hidden
   W2 shape: (32, 2) - Hidden → Output

📊 Test 1: Öğrenmeden Önce
   State: [ 0.1  0.5 -0.2  0.3]
   Q-Values (önce): [ 0.234 -0.123]
   Seçilen Action: 0

🎓 Test 2: Öğrenme - Action 0'ı Ödüllendirme
   Target Q-Values: [10.  -5.]
   Q-Values (sonra): [ 1.456 -0.789]
   Seçilen Action: 0
   ✅ Q-value değişimi:
      Action 0: 0.234 → 1.456 (artmalı)
      Action 1: -0.123 → -0.789 (azalmalı)

🔁 Test 3: 100 İterasyon Öğrenme
   Başlangıç Q-Values: [ 0.345 -0.234]
   İterasyon  20: Loss = 150.234567, Q = [ 0.456  5.234]
   İterasyon  40: Loss = 85.123456, Q = [ 0.523  10.123]
   İterasyon  60: Loss = 45.234567, Q = [ 0.567  14.456]
   İterasyon  80: Loss = 20.123456, Q = [ 0.589  17.234]
   İterasyon 100: Loss = 8.234567, Q = [ 0.601  18.789]
   Final Q-Values: [ 0.601  18.789]
   ✅ Action 1 Q-value: -0.234 → 18.789

📈 Loss grafiği kaydedildi: neural-network/learning_curve_q3.png

============================================================
✅ Tüm testler tamamlandı!
============================================================
```

**Gözlemler:**
1. ✅ Q-values target'a doğru hareket ediyor
2. ✅ Loss zamanla azalıyor
3. ✅ Network istediğimiz action'ı öğreniyor!

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `backward()` metodunu ekledim
- [ ] Loss gradient hesapladım: `dq = 2(predicted - target)`
- [ ] Output layer gradientleri hesapladım (dW2, db2)
- [ ] ReLU backprop yaptım: `dz1 = da1 * (z1 > 0)`
- [ ] Input layer gradientleri hesapladım (dW1, db1)
- [ ] Gradient descent ile ağırlıkları güncelledim
- [ ] Test 1: Tek iteration öğrenme çalışıyor
- [ ] Test 2: Çoklu iteration loss azalıyor
- [ ] Grafik oluşturdum (opsiyonel)

---

## 🎓 Ne Öğrendim?

1. **Loss Fonksiyonu**: MSE ile predicted ve target arasındaki farkı ölçtük
2. **Backpropagation**: Gradientleri geriye doğru hesapladık (chain rule)
3. **ReLU Backprop**: Sadece pozitif yerlerde gradient geçiyor
4. **Gradient Descent**: `W = W - α × ∇W` ile ağırlıkları güncelledik
5. **Öğrenme**: Loss azaldıkça Q-values target'a yaklaşıyor

---

## 🧠 Derinlemesine Sorular

1. **Learning rate çok büyük olursa ne olur?** (örn: 10.0)
2. **ReLU yerine sigmoid kullansaydık ne değişirdi?**
3. **Neden MSE? MAE (Mean Absolute Error) kullansak?**
4. **Batch size neden önemli?**

---

## 🚀 Sıradaki Görev

**Görev 4**: Experience Replay Buffer - Deneyimleri nasıl saklayacağız?

Q-Learning'de önemli bir trick! Hazır olduğunda devam edelim! 🎯
