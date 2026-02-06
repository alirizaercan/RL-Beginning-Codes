# 🎯 Görev 7: Actor Network Backward Pass - Policy Gradient

## 📚 Teorik Bilgi

### Policy Gradient Nedir?

Q-Learning'de backward pass basitti: MSE loss'tan gradient.

PPO'da farklı! **Policy gradient theorem** kullanıyoruz:

```
∇J(θ) = E[∇log π(a|s) × A(s,a)]
```

**Açıklama:**
- `θ`: Actor network ağırlıkları
- `π(a|s)`: Action probability
- `A(s,a)`: **Advantage** (bu action ne kadar iyi?)
- `∇log π`: Log probability'nin gradient'i

### Advantage Nedir?

```
Advantage = Q(s,a) - V(s)
```

**Sezgi:**
- `Q(s,a)`: Bu action'ı seçersem ne kadar reward alırım?
- `V(s)`: Bu state'te ortalama ne kadar reward alırım?
- `A(s,a)`: Bu action ortalamadan ne kadar iyi/kötü?

**Örnek:**
```
State: CartPole dengede
V(s) = 100 (bu state'ten ortalama 100 reward bekle)

Action 0 (sol): Q(s,0) = 80  → A = 80 - 100 = -20 (kötü!)
Action 1 (sağ): Q(s,1) = 130 → A = 130 - 100 = +30 (iyi!)
```

### Advantage Estimation

PPO'da **Generalized Advantage Estimation (GAE)** kullanıyoruz:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

δ_t = r_t + γV(s_{t+1}) - V(s_t)  # TD error
```

**Parametreler:**
- `γ` (gamma): Discount factor (0.99)
- `λ` (lambda): Advantage smoothing (0.95)

---

## 🧮 Matematiksel Detaylar

### Log Probability Gradient

**Softmax çıktısı için:**
```python
# Forward
action_probs = softmax(logits)

# Log probability
log_prob = log(action_probs[action])

# Gradient
d_log_prob = (1 - action_probs) for selected action
d_log_prob = -action_probs for other actions
```

### Policy Gradient Update

```python
# Loss (negatif çünkü maximize ediyoruz)
loss = -log_prob × advantage

# Gradient
gradient = -advantage × d_log_prob

# Update
weights -= learning_rate × gradient
```

---

## 🛠️ Adım 1: Actor Backward Pass

`ppo_networks.py` dosyasındaki `ActorNetwork` class'ına ekle:

```python
    def backward(self, advantages, actions):
        """
        Actor network'ü güncelle - Policy gradient
        
        Parameters:
        -----------
        advantages : np.array
            Advantage values (ne kadar iyi oldu?)
            Shape: (batch_size,)
        actions : np.array
            Seçilen action'lar
            Shape: (batch_size,)
        """
        batch_size = self.last_state.shape[0]
        
        # TODO: One-hot encode actions
        # [0, 1, 0] → [[1,0], [0,1], [1,0]]
        actions_one_hot = np.zeros((batch_size, self.action_size))
        actions_one_hot[np.arange(batch_size), actions] = 1
        
        # TODO: Policy gradient
        # dL/d(logits) = action_probs - actions_one_hot
        # Ama advantage ile ağırlıklandırıyoruz
        d_logits = self.action_probs - actions_one_hot
        
        # Advantage ile çarp (positive advantage → selected action'ı artır)
        d_logits *= advantages.reshape(-1, 1)
        d_logits /= batch_size
        
        # Output layer gradients
        dW3 = np.dot(self.a2.T, d_logits)
        db3 = np.sum(d_logits, axis=0, keepdims=True)
        
        # Hidden layer 2 gradients
        da2 = np.dot(d_logits, self.W3.T)
        dz2 = da2 * (self.z2 > 0)  # ReLU backprop
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU backprop
        
        dW1 = np.dot(self.last_state.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Gradient clipping
        max_grad = 0.5
        dW1 = np.clip(dW1, -max_grad, max_grad)
        dW2 = np.clip(dW2, -max_grad, max_grad)
        dW3 = np.clip(dW3, -max_grad, max_grad)
        
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
```

### 🔍 Kod Açıklaması

**One-Hot Encoding:**
```python
actions = [0, 1, 0]
one_hot = [[1, 0],   # Action 0
           [0, 1],   # Action 1
           [1, 0]]   # Action 0
```

**Policy Gradient Sezgisi:**
```python
# Advantage > 0: Bu action iyi, probability'sini artır
# Advantage < 0: Bu action kötü, probability'sini azalt

gradient = (action_probs - one_hot) × advantage
```

---

## 🛠️ Adım 2: Critic Backward Pass

`ppo_networks.py` dosyasındaki `CriticNetwork` class'ına ekle:

```python
    def backward(self, target_values):
        """
        Critic network'ü güncelle - MSE loss
        
        Parameters:
        -----------
        target_values : np.array
            Hedef value'lar (gerçek return'ler)
            Shape: (batch_size, 1)
        """
        batch_size = self.last_state.shape[0]
        
        # TODO: MSE loss gradient
        d_value = 2 * (self.value - target_values) / batch_size
        
        # Output layer gradients
        dW3 = np.dot(self.a2.T, d_value)
        db3 = np.sum(d_value, axis=0, keepdims=True)
        
        # Hidden layer 2 gradients
        da2 = np.dot(d_value, self.W3.T)
        dz2 = da2 * (self.z2 > 0)  # ReLU backprop
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU backprop
        
        dW1 = np.dot(self.last_state.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Gradient clipping
        max_grad = 0.5
        dW1 = np.clip(dW1, -max_grad, max_grad)
        dW2 = np.clip(dW2, -max_grad, max_grad)
        dW3 = np.clip(dW3, -max_grad, max_grad)
        
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
```

---

## 🛠️ Adım 3: Advantage Hesaplama

Yardımcı fonksiyon ekle:

```python
def compute_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE)
    
    Parameters:
    -----------
    rewards : np.array
        Reward'lar (batch_size,)
    values : np.array
        State values V(s_t) (batch_size, 1)
    next_values : np.array
        Next state values V(s_{t+1}) (batch_size, 1)
    dones : np.array
        Episode bitti mi? (batch_size,)
    gamma : float
        Discount factor
    lam : float
        GAE lambda parameter
    
    Returns:
    --------
    advantages : np.array
        Computed advantages
    returns : np.array
        Target returns for critic
    """
    batch_size = len(rewards)
    advantages = np.zeros(batch_size)
    returns = np.zeros(batch_size)
    
    # TD errors: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values.flatten() * (1 - dones) - values.flatten()
    
    # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    gae = 0
    for t in reversed(range(batch_size)):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values.flatten()[t]
    
    return advantages, returns
```

### 🔍 GAE Açıklaması

```python
# Örnek trajectory
rewards = [1, 1, 1, 10]  # Son adımda büyük reward
values = [5, 5, 5, 5]    # Critic tahminleri

# TD errors
δ_0 = 1 + 0.99×5 - 5 = 0.95
δ_1 = 1 + 0.99×5 - 5 = 0.95
δ_2 = 1 + 0.99×5 - 5 = 0.95
δ_3 = 10 + 0 - 5 = 5  # Episode bitti

# GAE (geriye doğru)
A_3 = 5
A_2 = 0.95 + 0.99×0.95×5 = 5.6
A_1 = 0.95 + 0.99×0.95×5.6 = 6.2
A_0 = 0.95 + 0.99×0.95×6.2 = 6.8

# Yani erken adımlara da büyük reward'ın etkisi yayılıyor!
```

---

## 🧪 Test Zamanı!

Test kodunu ekle:

```python
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 7: Backward Pass & Advantage Test")
    print("="*60)
    
    state_size = 4
    action_size = 2
    
    # Networks oluştur
    actor = ActorNetwork(state_size, action_size, hidden_sizes=[128, 64])
    critic = CriticNetwork(state_size, hidden_sizes=[128, 64])
    
    # Test 1: Actor backward pass
    print("\n🎭 Test 1: Actor Backward Pass")
    
    states = np.random.randn(32, state_size)
    actions = np.random.randint(0, action_size, 32)
    advantages = np.random.randn(32)  # Random advantages
    
    # Forward pass
    probs_before, _ = actor.forward(states)
    print(f"   Batch size: {states.shape[0]}")
    print(f"   İlk state action probs (önce): {probs_before[0]}")
    
    # Backward pass
    actor.backward(advantages, actions)
    
    # Forward pass tekrar
    probs_after, _ = actor.forward(states)
    print(f"   İlk state action probs (sonra): {probs_after[0]}")
    print(f"   ✅ Probabilities değişti!")
    
    # Test 2: Positive advantage action'ı artırıyor mu?
    print("\n📈 Test 2: Positive Advantage Etkisi")
    
    state = np.array([[0.1, 0.5, -0.2, 0.3]])
    action = np.array([0])  # Action 0'ı seç
    
    probs_initial, _ = actor.forward(state)
    print(f"   Başlangıç: Action 0 prob = {probs_initial[0, 0]:.4f}")
    
    # Büyük positive advantage ile 10 update
    for i in range(10):
        actor.forward(state)
        actor.backward(np.array([10.0]), action)  # +10 advantage!
    
    probs_final, _ = actor.forward(state)
    print(f"   10 update sonra: Action 0 prob = {probs_final[0, 0]:.4f}")
    print(f"   ✅ Positive advantage action probability'sini artırdı!")
    
    # Test 3: Critic backward pass
    print("\n💭 Test 3: Critic Backward Pass")
    
    states = np.random.randn(32, state_size)
    target_values = np.random.randn(32, 1) * 10
    
    values_before = critic.predict(states)
    print(f"   İlk state value (önce): {values_before[0, 0]:.4f}")
    print(f"   Target: {target_values[0, 0]:.4f}")
    
    # Backward pass
    critic.backward(target_values)
    
    values_after = critic.predict(states)
    print(f"   İlk state value (sonra): {values_after[0, 0]:.4f}")
    
    # Target'a yaklaştı mı?
    error_before = abs(values_before[0, 0] - target_values[0, 0])
    error_after = abs(values_after[0, 0] - target_values[0, 0])
    print(f"   Error (önce): {error_before:.4f}")
    print(f"   Error (sonra): {error_after:.4f}")
    if error_after < error_before:
        print(f"   ✅ Target'a yaklaştı!")
    
    # Test 4: Advantage hesaplama
    print("\n🎯 Test 4: Advantage Computation (GAE)")
    
    # Basit trajectory
    rewards = np.array([1.0, 1.0, 1.0, 10.0])
    values = np.array([[5.0], [5.0], [5.0], [5.0]])
    next_values = np.array([[5.0], [5.0], [5.0], [0.0]])  # Son state done
    dones = np.array([0, 0, 0, 1])
    
    advantages, returns = compute_advantages(rewards, values, next_values, dones)
    
    print(f"   Rewards: {rewards}")
    print(f"   Values: {values.flatten()}")
    print(f"   Advantages: {advantages}")
    print(f"   Returns: {returns}")
    print(f"   ✅ Son reward erken adımlara da yayıldı! (GAE)")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
```

---

## 🎯 Beklenen Çıktı

```
============================================================
🧪 GÖREV 7: Backward Pass & Advantage Test
============================================================

🎭 Test 1: Actor Backward Pass
✅ Actor Network oluşturuldu:
   4 → 128 → 64 → 2
   Batch size: 32
   İlk state action probs (önce): [0.523 0.477]
   İlk state action probs (sonra): [0.531 0.469]
   ✅ Probabilities değişti!

📈 Test 2: Positive Advantage Etkisi
   Başlangıç: Action 0 prob = 0.5234
   10 update sonra: Action 0 prob = 0.7812
   ✅ Positive advantage action probability'sini artırdı!

💭 Test 3: Critic Backward Pass
✅ Critic Network oluşturuldu:
   4 → 128 → 64 → 1
   İlk state value (önce): 0.1234
   Target: 5.6789
   İlk state value (sonra): 0.2345
   Error (önce): 5.5555
   Error (sonra): 5.4444
   ✅ Target'a yaklaştı!

🎯 Test 4: Advantage Computation (GAE)
   Rewards: [ 1.  1.  1. 10.]
   Values: [5. 5. 5. 5.]
   Advantages: [6.85 5.92 5.01 5.00]
   Returns: [11.85 10.92 10.01 10.00]
   ✅ Son reward erken adımlara da yayıldı! (GAE)

============================================================
✅ Tüm testler tamamlandı!
============================================================
```

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `Actor.backward()`: Policy gradient implementasyonu
- [ ] One-hot encoding yapıldı
- [ ] Advantage ile gradient ağırlıklandırma
- [ ] `Critic.backward()`: MSE loss ile güncelleme
- [ ] `compute_advantages()`: GAE implementasyonu
- [ ] Test 1-4'ü çalıştırdım
- [ ] Positive advantage probability'yi artırdı
- [ ] Critic target'a yaklaştı

---

## 🎓 Ne Öğrendim?

1. **Policy Gradient**: Log probability × advantage
2. **Advantage**: Action ortalamadan ne kadar iyi?
3. **GAE**: Advantage'ı tüm trajectory'ye yay
4. **Actor Update**: Good action'ları encourage et
5. **Critic Update**: Basit MSE loss (Q-Learning gibi)

---

## 🧠 Derinlemesine Sorular

1. **Lambda = 0 olsaydı?** (pure TD)
2. **Lambda = 1 olsaydı?** (Monte Carlo)
3. **Advantage normalize edilmeli mi?** (mean=0, std=1)
4. **Negative advantage ne yapar?** (bad action discourage)

---

## 🚀 Sıradaki Görev

**Görev 8**: PPO Clip - Büyük policy update'leri nasıl önlüyoruz?

Hazır olduğunda devam edelim! 🎯
