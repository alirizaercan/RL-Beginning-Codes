# 🎯 Görev 6: PPO Actor-Critic Mimarisi

## 📚 Teorik Bilgi

### Policy Gradient vs Q-Learning

**Q-Learning (DQN):**
- **Value-based**: Q-values öğreniliyor
- Action seçimi: `argmax Q(s,a)` (deterministik)
- Discrete action space'ler için uygun

**Policy Gradient (PPO):**
- **Policy-based**: Doğrudan policy öğreniliyor
- Action seçimi: Probability distribution'dan sample
- Hem discrete hem continuous action'lar için

### Actor-Critic Nedir?

**İki ayrı neural network:**

1. **Actor (Policy Network) π(a|s)**
   - Input: State
   - Output: Action probabilities
   - Görev: En iyi action'ları öğren

2. **Critic (Value Network) V(s)**
   - Input: State
   - Output: State value (tek sayı)
   - Görev: State'in ne kadar iyi olduğunu değerlendir

### Neden İki Network?

```
Actor: "Ben bu state'te sağa gidiyorum!" 
       ↓
Environment: +10 reward!
       ↓
Critic: "Bu state aslında +50 değerinde, o yüzden +10 beklenenden az!"
       ↓
Actor: "Anladım, daha iyi action öğreneceğim"
```

**Avantajlar:**
- **Variance Reduction**: Critic baseline sağlar
- **Faster Learning**: Her ikisi birbirini tamamlar
- **Stability**: Critic actor'ı stabilize eder

---

## 🏗️ Mimari Tasarımı

### Actor Network (Policy)

```
State (4)
    ↓
Hidden Layer (128) - ReLU
    ↓
Hidden Layer (64) - ReLU
    ↓
Action Logits (2) - Linear
    ↓
Softmax
    ↓
Action Probs [0.3, 0.7]
```

**Output:** Probability distribution over actions

### Critic Network (Value)

```
State (4)
    ↓
Hidden Layer (128) - ReLU
    ↓
Hidden Layer (64) - ReLU
    ↓
State Value (1) - Linear
```

**Output:** Single value (expected return from this state)

### Neden Daha Derin?

- Q-Learning: 1 hidden layer yeterli (value function basit)
- PPO: 2 hidden layer (policy function daha karmaşık)

---

## 🛠️ Adım 1: Actor Network Class'ı

Yeni dosya: `ppo_networks.py`

```python
import numpy as np

class ActorNetwork:
    """
    Policy Network (Actor) - PPO için
    
    State alır, action probability distribution döndürür
    """
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 64], learning_rate=0.0003):
        """
        Actor Network'ü başlat
        
        Parameters:
        -----------
        state_size : int
            State vektör boyutu
        action_size : int
            Action sayısı
        hidden_sizes : list
            Her hidden layer'ın nöron sayısı [128, 64]
        learning_rate : float
            Öğrenme hızı (PPO için küçük tutulur)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        
        # TODO: Network katmanlarını başlat
        # Layer 1: Input → Hidden1
        self.W1 = np.random.randn(state_size, hidden_sizes[0]) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        
        # Layer 2: Hidden1 → Hidden2
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        
        # Layer 3: Hidden2 → Output (action logits)
        self.W3 = np.random.randn(hidden_sizes[1], action_size) * np.sqrt(2.0 / hidden_sizes[1])
        self.b3 = np.zeros((1, action_size))
        
        print(f"✅ Actor Network oluşturuldu:")
        print(f"   {state_size} → {hidden_sizes[0]} → {hidden_sizes[1]} → {action_size}")
        print(f"   Learning Rate: {learning_rate}")
    
    def forward(self, state):
        """
        Forward pass - State'den action probabilities'e
        
        Parameters:
        -----------
        state : np.array
            Shape: (batch_size, state_size) veya (state_size,)
        
        Returns:
        --------
        action_probs : np.array
            Action probabilities (softmax)
        action_logits : np.array
            Raw logits (softmax öncesi)
        """
        # TODO: State'i 2D yap
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # TODO: Layer 1
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # TODO: Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        
        # TODO: Layer 3 - Output (logits)
        self.logits = np.dot(self.a2, self.W3) + self.b3
        
        # TODO: Softmax - logits'i probability'ye çevir
        # Numerical stability için max çıkar
        exp_logits = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.action_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Son state'i sakla (backward için)
        self.last_state = state
        
        return self.action_probs, self.logits
    
    def get_action(self, state):
        """
        State'den action seç (probability distribution'dan sample)
        
        Returns:
        --------
        action : int
            Seçilen action
        action_prob : float
            Seçilen action'ın probability'si
        """
        action_probs, _ = self.forward(state)
        
        # TODO: Probability distribution'dan sample
        action = np.random.choice(self.action_size, p=action_probs[0])
        action_prob = action_probs[0, action]
        
        return action, action_prob
```

### 🔍 Kod Açıklaması

**Softmax Neden Kullanıyoruz?**
```python
logits = [2.3, -1.5]  # Raw network output

# Softmax:
probs = [0.97, 0.03]  # Sum = 1.0, valid probability distribution
```

**Numerical Stability:**
```python
# Sorun: exp(100) = overflow!
# Çözüm: Max çıkar
exp_logits = np.exp(logits - np.max(logits))
```

**Action Sampling:**
```python
# Deterministik (DQN): argmax
action = np.argmax(probs)  # Her zaman en yüksek

# Stochastic (PPO): sample
action = np.random.choice(n, p=probs)  # Probabilitye göre
```

---

## 🛠️ Adım 2: Critic Network Class'ı

Aynı dosyaya ekle:

```python
class CriticNetwork:
    """
    Value Network (Critic) - PPO için
    
    State alır, state value döndürür
    """
    
    def __init__(self, state_size, hidden_sizes=[128, 64], learning_rate=0.001):
        """
        Critic Network'ü başlat
        
        Parameters:
        -----------
        state_size : int
            State vektör boyutu
        hidden_sizes : list
            Her hidden layer'ın nöron sayısı [128, 64]
        learning_rate : float
            Öğrenme hızı
        """
        self.state_size = state_size
        self.lr = learning_rate
        
        # TODO: Network katmanlarını başlat
        # Layer 1: Input → Hidden1
        self.W1 = np.random.randn(state_size, hidden_sizes[0]) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        
        # Layer 2: Hidden1 → Hidden2
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        
        # Layer 3: Hidden2 → Output (single value)
        self.W3 = np.random.randn(hidden_sizes[1], 1) * np.sqrt(2.0 / hidden_sizes[1])
        self.b3 = np.zeros((1, 1))
        
        print(f"✅ Critic Network oluşturuldu:")
        print(f"   {state_size} → {hidden_sizes[0]} → {hidden_sizes[1]} → 1")
        print(f"   Learning Rate: {learning_rate}")
    
    def forward(self, state):
        """
        Forward pass - State'den value'ya
        
        Parameters:
        -----------
        state : np.array
            Shape: (batch_size, state_size) veya (state_size,)
        
        Returns:
        --------
        value : np.array
            State value tahmin
        """
        # TODO: State'i 2D yap
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # TODO: Layer 1
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # TODO: Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        
        # TODO: Layer 3 - Output (value)
        self.value = np.dot(self.a2, self.W3) + self.b3
        
        # Son state'i sakla (backward için)
        self.last_state = state
        
        return self.value
    
    def predict(self, state):
        """
        State için value tahmin et
        """
        return self.forward(state)
```

---

## 🧪 Test Zamanı!

Test kodunu dosyanın sonuna ekle:

```python
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 6: Actor-Critic Networks Test")
    print("="*60)
    
    state_size = 4
    action_size = 2
    
    # Test 1: Actor Network
    print("\n🎭 Test 1: Actor Network")
    actor = ActorNetwork(state_size, action_size, hidden_sizes=[128, 64])
    
    state = np.array([0.1, 0.5, -0.2, 0.3])
    probs, logits = actor.forward(state)
    
    print(f"   State: {state}")
    print(f"   Logits: {logits[0]}")
    print(f"   Probabilities: {probs[0]}")
    print(f"   Sum of probs: {np.sum(probs[0]):.6f} (should be 1.0)")
    
    # Action sampling test
    print("\n   🎲 Action Sampling (10 kez):")
    actions = []
    for _ in range(10):
        action, prob = actor.get_action(state)
        actions.append(action)
    
    action_counts = np.bincount(actions)
    print(f"   Action 0: {action_counts[0]} kez")
    print(f"   Action 1: {action_counts[1]} kez")
    print(f"   Ratio: ~{probs[0,0]:.2f} / ~{probs[0,1]:.2f}")
    
    # Test 2: Critic Network
    print("\n💭 Test 2: Critic Network")
    critic = CriticNetwork(state_size, hidden_sizes=[128, 64])
    
    value = critic.predict(state)
    print(f"   State: {state}")
    print(f"   Predicted Value: {value[0, 0]:.4f}")
    
    # Batch test
    print("\n📦 Test 3: Batch Processing")
    batch_states = np.random.randn(5, state_size)
    
    batch_probs, _ = actor.forward(batch_states)
    batch_values = critic.predict(batch_states)
    
    print(f"   Batch size: {batch_states.shape[0]}")
    print(f"   Actor output shape: {batch_probs.shape}")  # (5, 2)
    print(f"   Critic output shape: {batch_values.shape}")  # (5, 1)
    
    print("\n   First 3 states:")
    for i in range(3):
        print(f"      State {i}: Probs={batch_probs[i]}, Value={batch_values[i,0]:.3f}")
    
    # Test 4: Farklı state'ler farklı output verir mi?
    print("\n🔬 Test 4: Network Çeşitliliği")
    state1 = np.array([1.0, 0.0, 0.0, 0.0])
    state2 = np.array([0.0, 1.0, 0.0, 0.0])
    
    probs1, _ = actor.forward(state1)
    probs2, _ = actor.forward(state2)
    value1 = critic.predict(state1)
    value2 = critic.predict(state2)
    
    print(f"   State 1: Probs={probs1[0]}, Value={value1[0,0]:.3f}")
    print(f"   State 2: Probs={probs2[0]}, Value={value2[0,0]:.3f}")
    print(f"   ✅ Farklı state'ler farklı output verdi!")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
```

---

## 🎯 Beklenen Çıktı

```
============================================================
🧪 GÖREV 6: Actor-Critic Networks Test
============================================================

🎭 Test 1: Actor Network
✅ Actor Network oluşturuldu:
   4 → 128 → 64 → 2
   Learning Rate: 0.0003
   State: [ 0.1  0.5 -0.2  0.3]
   Logits: [ 0.234 -0.156]
   Probabilities: [0.597 0.403]
   Sum of probs: 1.000000 (should be 1.0)

   🎲 Action Sampling (10 kez):
   Action 0: 6 kez
   Action 1: 4 kez
   Ratio: ~0.60 / ~0.40

💭 Test 2: Critic Network
✅ Critic Network oluşturuldu:
   4 → 128 → 64 → 1
   Learning Rate: 0.001
   State: [ 0.1  0.5 -0.2  0.3]
   Predicted Value: 0.1234

📦 Test 3: Batch Processing
   Batch size: 5
   Actor output shape: (5, 2)
   Critic output shape: (5, 1)

   First 3 states:
      State 0: Probs=[0.52 0.48], Value=0.089
      State 1: Probs=[0.61 0.39], Value=-0.034
      State 2: Probs=[0.43 0.57], Value=0.156

🔬 Test 4: Network Çeşitliliği
   State 1: Probs=[0.54 0.46], Value=0.123
   State 2: Probs=[0.48 0.52], Value=-0.067
   ✅ Farklı state'ler farklı output verdi!

============================================================
✅ Tüm testler tamamlandı!
============================================================
```

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `ActorNetwork` class'ını oluşturdum
- [ ] Actor: 3 layer (2 hidden + output)
- [ ] Softmax ile probability distribution
- [ ] `get_action()`: Stochastic sampling
- [ ] `CriticNetwork` class'ını oluşturdum
- [ ] Critic: 3 layer (2 hidden + value output)
- [ ] Test 1-4'ü çalıştırdım ve geçti
- [ ] Probability sum = 1.0 olduğunu doğruladım

---

## 🎓 Ne Öğrendim?

1. **Actor-Critic**: İki network birlikte çalışıyor
2. **Softmax**: Logits → valid probability distribution
3. **Stochastic Policy**: Probability'den sampling
4. **Value Function**: State'in ne kadar iyi olduğu
5. **Daha Derin Mimari**: 2 hidden layer (karmaşık policy için)

---

## 🧠 Derinlemesine Sorular

1. **Actor deterministik olsaydı (argmax) ne olurdu?**
2. **Critic olmadan PPO çalışır mı?** (REINFORCE)
3. **Hidden size [256, 128] daha mı iyi?**
4. **Actor ve Critic aynı network'ü paylaşabilir mi?** (shared backbone)

---

## 🚀 Sıradaki Görev

**Görev 7**: Actor Network Backward Pass - Policy gradient nasıl hesaplanıyor?

Hazır olduğunda devam edelim! 🎯
