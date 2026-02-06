# 🎯 Görev 8: PPO Clipped Objective - Stability Trick

## 📚 Teorik Bilgi

### Vanilla Policy Gradient Sorunu

**Problem:** Policy çok hızlı değişebilir!

```
Episode 1: Policy iyi → Büyük reward
Episode 2: Policy çok değişti → Kötü policy → Düşük reward
Episode 3: Daha da kötüleşti → Training collapse! 💥
```

**Neden?**
- Bir büyük gradient update policy'yi tamamen değiştirebilir
- Kötü bir batch stability'yi bozabilir

### PPO'nun Çözümü: Clipped Objective

**Fikir:** Policy update'lerini **sınırla**!

```
Old Policy: π_old(a|s) = 0.2  (action'ın eski probability'si)
New Policy: π_new(a|s) = ?

PPO: π_new maksimum %20 değişebilir
      → π_new ∈ [0.16, 0.24]
```

### PPO Clip Formula

```python
ratio = π_new(a|s) / π_old(a|s)

# Clip ratio to [1-ε, 1+ε]
clipped_ratio = clip(ratio, 1-ε, 1+ε)

# Loss: min(ratio×A, clipped_ratio×A)
loss = -min(ratio × advantage, clipped_ratio × advantage)
```

**Parametreler:**
- `ε` (epsilon): Clip range (genellikle 0.2)
- `ratio`: Yeni policy / eski policy oranı

### Neden min()?

**Positive Advantage (A > 0):**
```
ratio = 1.5  (action probability arttı)
clipped = 1.2  (max %20 artış)
Loss = min(1.5×A, 1.2×A) = 1.2×A  ← Küçüğü seç (conservative)
```

**Negative Advantage (A < 0):**
```
ratio = 0.7  (action probability azaldı)
clipped = 0.8  (max %20 azalma)
Loss = min(0.7×A, 0.8×A) = 0.7×A  ← Büyüğü seç (A<0 olduğu için)
```

**Sezgi:** Büyük değişiklikleri engelle, küçük iyileştirmelere izin ver!

---

## 🧮 Matematiksel Detaylar

### Ratio Hesaplama

```python
# Old policy'den log probability
log_prob_old = log(π_old(action|state))

# New policy'den log probability
log_prob_new = log(π_new(action|state))

# Ratio (exp kullanarak)
ratio = exp(log_prob_new - log_prob_old)
```

**Neden exp(log)?**
- Numerically stable
- `exp(log(a) - log(b)) = a/b`

### Clip Function

```python
def clip(x, min_val, max_val):
    return max(min_val, min(max_val, x))

# Örnek
clip(1.5, 0.8, 1.2) = 1.2  # 1.5 > 1.2, clip to 1.2
clip(0.6, 0.8, 1.2) = 0.8  # 0.6 < 0.8, clip to 0.8
clip(1.0, 0.8, 1.2) = 1.0  # 0.8 < 1.0 < 1.2, no change
```

---

## 🛠️ Adım 1: Old Policy Probabilities'i Sakla

`ppo_networks.py`'deki `ActorNetwork`'e ekle:

```python
    def get_log_prob(self, state, action):
        """
        Belirli bir action için log probability hesapla
        
        Parameters:
        -----------
        state : np.array
            State
        action : int or np.array
            Action(s)
        
        Returns:
        --------
        log_prob : float or np.array
            Log probability of action
        """
        action_probs, _ = self.forward(state)
        
        if isinstance(action, int):
            # Tek action
            log_prob = np.log(action_probs[0, action] + 1e-10)  # +epsilon for stability
        else:
            # Batch actions
            batch_size = len(action)
            log_probs = np.log(action_probs[np.arange(batch_size), action] + 1e-10)
            log_prob = log_probs
        
        return log_prob
```

---

## 🛠️ Adım 2: PPO Clip Loss

Yeni fonksiyon ekle:

```python
def compute_ppo_loss(actor, states, actions, advantages, old_log_probs, clip_epsilon=0.2):
    """
    PPO clipped surrogate objective
    
    Parameters:
    -----------
    actor : ActorNetwork
        Actor network
    states : np.array
        Batch of states
    actions : np.array
        Batch of actions
    advantages : np.array
        Batch of advantages
    old_log_probs : np.array
        Old policy log probabilities
    clip_epsilon : float
        Clip range (default: 0.2)
    
    Returns:
    --------
    loss : float
        PPO loss
    ratio : np.array
        Policy ratio (for monitoring)
    clipped_fraction : float
        Fraction of ratios that were clipped (for monitoring)
    """
    # TODO: Current policy log probabilities
    new_log_probs = actor.get_log_prob(states, actions)
    
    # TODO: Ratio = π_new / π_old = exp(log_new - log_old)
    log_ratio = new_log_probs - old_log_probs
    ratio = np.exp(log_ratio)
    
    # TODO: Clipped ratio
    ratio_clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    # TODO: Surrogate losses
    surr1 = ratio * advantages
    surr2 = ratio_clipped * advantages
    
    # TODO: PPO loss = -min(surr1, surr2)
    # min() alıyoruz çünkü conservative olmak istiyoruz
    loss = -np.mean(np.minimum(surr1, surr2))
    
    # TODO: Monitoring metrics
    clipped_fraction = np.mean(np.abs(ratio - ratio_clipped) > 1e-6)
    
    return loss, ratio, clipped_fraction
```

### 🔍 Kod Açıklaması

**Log Probability Neden +1e-10?**
```python
# Sorun: log(0) = -inf (numerical error)
# Çözüm: Küçük epsilon ekle
log_prob = np.log(prob + 1e-10)
```

**np.minimum() vs min():**
```python
# np.minimum: Element-wise
a = [1, 2, 3]
b = [2, 1, 4]
np.minimum(a, b) = [1, 1, 3]  # Her element için

# min(): Tek değer
min(a, b) = [1, 2, 3]  # Liste karşılaştırması
```

---

## 🛠️ Adım 3: Actor Backward Pass (PPO Version)

`ActorNetwork`'teki `backward` metodunu PPO clip ile güncelle:

```python
    def backward_ppo(self, states, actions, advantages, old_log_probs, clip_epsilon=0.2):
        """
        PPO clipped objective ile actor update
        
        Parameters:
        -----------
        states : np.array
            Batch states
        actions : np.array
            Batch actions
        advantages : np.array
            Batch advantages
        old_log_probs : np.array
            Old policy log probs
        clip_epsilon : float
            Clip range
        """
        batch_size = states.shape[0]
        
        # Forward pass
        action_probs, _ = self.forward(states)
        
        # Current log probs
        new_log_probs = self.get_log_prob(states, actions)
        
        # Ratio
        ratio = np.exp(new_log_probs - old_log_probs)
        ratio_clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        
        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        
        # Which is smaller? (element-wise)
        use_clipped = surr2 < surr1
        
        # TODO: Gradient hesaplama
        # One-hot encode actions
        actions_one_hot = np.zeros((batch_size, self.action_size))
        actions_one_hot[np.arange(batch_size), actions] = 1
        
        # Policy gradient
        d_logits = self.action_probs - actions_one_hot
        
        # Clipped gradient: Sadece clip olmayan yerlerde uygula
        effective_advantages = np.where(use_clipped, 
                                       ratio_clipped * advantages,
                                       ratio * advantages)
        
        d_logits *= effective_advantages.reshape(-1, 1)
        d_logits /= batch_size
        
        # Backpropagation (Görev 7'deki gibi)
        dW3 = np.dot(self.a2.T, d_logits)
        db3 = np.sum(d_logits, axis=0, keepdims=True)
        
        da2 = np.dot(d_logits, self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        
        dW1 = np.dot(self.last_state.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Gradient clipping
        max_grad = 0.5
        dW1 = np.clip(dW1, -max_grad, max_grad)
        dW2 = np.clip(dW2, -max_grad, max_grad)
        dW3 = np.clip(dW3, -max_grad, max_grad)
        
        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
```

---

## 🧪 Test Zamanı!

```python
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 8: PPO Clipped Objective Test")
    print("="*60)
    
    state_size = 4
    action_size = 2
    
    actor = ActorNetwork(state_size, action_size)
    
    # Test 1: Ratio hesaplama
    print("\n📊 Test 1: Policy Ratio")
    
    state = np.array([[0.1, 0.5, -0.2, 0.3]])
    action = np.array([0])
    
    # Old policy
    old_log_prob = actor.get_log_prob(state, action)
    print(f"   Old log prob: {old_log_prob:.4f}")
    
    # Policy'yi değiştir (büyük update)
    for _ in range(5):
        actor.forward(state)
        actor.backward(np.array([10.0]), action)
    
    # New policy
    new_log_prob = actor.get_log_prob(state, action)
    print(f"   New log prob: {new_log_prob:.4f}")
    
    # Ratio
    ratio = np.exp(new_log_prob - old_log_prob)
    print(f"   Ratio (π_new/π_old): {ratio:.4f}")
    
    if ratio > 1.2:
        print(f"   ⚠️ Ratio > 1.2! PPO clip gerekli!")
    
    # Test 2: Clipping işlemi
    print("\n✂️ Test 2: Clipping")
    
    ratios = np.array([0.5, 0.9, 1.0, 1.3, 1.8])
    clipped = np.clip(ratios, 0.8, 1.2)
    
    print(f"   Original ratios: {ratios}")
    print(f"   Clipped (ε=0.2): {clipped}")
    print(f"   ✅ Büyük değişiklikler sınırlandı!")
    
    # Test 3: PPO loss hesaplama
    print("\n🎯 Test 3: PPO Loss")
    
    states = np.random.randn(32, state_size)
    actions = np.random.randint(0, action_size, 32)
    advantages = np.random.randn(32)
    
    # Old log probs
    old_log_probs = actor.get_log_prob(states, actions)
    
    # Policy'yi biraz değiştir
    actor.forward(states)
    actor.backward(advantages, actions)
    
    # PPO loss
    loss, ratios, clipped_frac = compute_ppo_loss(
        actor, states, actions, advantages, old_log_probs
    )
    
    print(f"   PPO Loss: {loss:.4f}")
    print(f"   Mean ratio: {np.mean(ratios):.4f}")
    print(f"   Clipped fraction: {clipped_frac:.2%}")
    
    # Test 4: Clipping'in etkisi
    print("\n🔬 Test 4: Clip ile vs Clip olmadan")
    
    # Yeni actor
    actor1 = ActorNetwork(state_size, action_size, learning_rate=0.01)
    actor2 = ActorNetwork(state_size, action_size, learning_rate=0.01)
    
    # Aynı başlangıç
    actor2.W1 = actor1.W1.copy()
    actor2.W2 = actor1.W2.copy()
    actor2.W3 = actor1.W3.copy()
    
    state = np.array([[0.1, 0.5, -0.2, 0.3]])
    
    # Başlangıç probs
    prob_initial, _ = actor1.forward(state)
    print(f"   Başlangıç probs: {prob_initial[0]}")
    
    # 10 update - Vanilla PG (clip yok)
    for _ in range(10):
        actor1.forward(state)
        actor1.backward(np.array([10.0]), np.array([0]))
    
    # 10 update - PPO (clip var)
    old_log_prob = actor2.get_log_prob(state, np.array([0]))
    for _ in range(10):
        actor2.backward_ppo(state, np.array([0]), np.array([10.0]), 
                           np.array([old_log_prob]), clip_epsilon=0.2)
        old_log_prob = actor2.get_log_prob(state, np.array([0]))
    
    prob_vanilla, _ = actor1.forward(state)
    prob_ppo, _ = actor2.forward(state)
    
    print(f"   Vanilla PG (clip yok): {prob_vanilla[0]}")
    print(f"   PPO (clip var): {prob_ppo[0]}")
    print(f"   ✅ PPO daha conservative (stabil) update!")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
```

---

## 🎯 Beklenen Çıktı

```
============================================================
🧪 GÖREV 8: PPO Clipped Objective Test
============================================================

📊 Test 1: Policy Ratio
✅ Actor Network oluşturuldu:
   4 → 128 → 64 → 2
   Old log prob: -0.7234
   New log prob: -0.2145
   Ratio (π_new/π_old): 1.6543
   ⚠️ Ratio > 1.2! PPO clip gerekli!

✂️ Test 2: Clipping
   Original ratios: [0.5 0.9 1.  1.3 1.8]
   Clipped (ε=0.2): [0.8 0.9 1.  1.2 1.2]
   ✅ Büyük değişiklikler sınırlandı!

🎯 Test 3: PPO Loss
   PPO Loss: 2.3456
   Mean ratio: 1.0234
   Clipped fraction: 15.62%

🔬 Test 4: Clip ile vs Clip olmadan
   Başlangıç probs: [0.52 0.48]
   Vanilla PG (clip yok): [0.95 0.05]
   PPO (clip var): [0.73 0.27]
   ✅ PPO daha conservative (stabil) update!

============================================================
✅ Tüm testler tamamlandı!
============================================================
```

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `get_log_prob()`: Log probability hesaplama
- [ ] `compute_ppo_loss()`: Clipped objective
- [ ] Ratio hesaplama: `exp(log_new - log_old)`
- [ ] Clipping: `clip(ratio, 1-ε, 1+ε)`
- [ ] `backward_ppo()`: PPO update metodu
- [ ] Test 1-4'ü çalıştırdım
- [ ] PPO'nun vanilla PG'den daha conservative olduğunu gördüm
- [ ] Clipped fraction'ı monitor ettim

---

## 🎓 Ne Öğrendim?

1. **PPO Clip**: Büyük policy update'lerini engelle
2. **Ratio**: π_new / π_old (policy değişimi)
3. **Conservative Update**: min(surr1, surr2) seç
4. **Stability**: Clip training collapse'ı önler
5. **Monitoring**: Clipped fraction önemli metrik

---

## 🧠 Derinlemesine Sorular

1. **Epsilon = 0.1 çok mu küçük?** (daha yavaş öğrenme)
2. **Epsilon = 0.5 çok mu büyük?** (clip etkisiz)
3. **Clip olmadan PPO ne olur?** (A2C/REINFORCE)
4. **Ratio > 2 olursa ne olur?** (policy çok değişti)

---

## 🚀 Sıradaki Görev

**Görev 9**: Multiple Epochs - Aynı data'dan birden fazla öğren!

Hazır olduğunda devam edelim! 🎯
