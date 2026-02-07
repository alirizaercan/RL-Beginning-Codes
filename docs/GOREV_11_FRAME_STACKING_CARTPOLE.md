# 🎬 Görev 11: Frame Stacking ile CartPole PPO

## 🎯 Bu Görevin Amacı

CartPole oyununda frame stacking tekniğini kullanarak PPO agent'inin performansını karşılaştırmak.

### Neden Frame Stacking CartPole'de?

**Normal CartPole State:**
```
[x, y, θ, ω]  → 4 sayısal değer
→ Zaten tam bilgi var!
```

**Frame Stacking CartPole (Visual):**
```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Frame 1 │ │ Frame 2 │ │ Frame 3 │ │ Frame 4 │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
    84x84      84x84      84x84      84x84
    
→ Zamansal bilgi (velocity) pixels'dan çıkarılıyor
→ Agent gerçekten hareket yönünü görüyor!
```

---

## 📚 Teori: Frame Stacking Nedir?

### Basit Açıklama

```
❌ Tek Frame (statik):
   ▯─────────▯
   │         │
   └─────────┘
   
   → "Çubuk hangi tarafa yağılıyor?" Bilmiyorum!

✅ 4 Stacked Frame (dinamik):
   Frame t-3  Frame t-2  Frame t-1  Frame t (now)
   ▯────      ▯──────    ▯────────  ▯──────────
   
   → Çubuk SĞA doğru hareket ediyor! (clear motion)
```

### Matematiksel Tanım

**Deque (Double-Ended Queue) Kullanarak:**

```
Step 0: Reset
  frames = [empty, empty, empty, empty]

Step 1: İlk observation gelir
  frame_1 = preprocess(obs)
  frames = [frame_1, frame_1, frame_1, frame_1]  (doldur)

Step 2: Action yap
  frame_2 = preprocess(obs)
  frames = [frame_1, frame_1, frame_1, frame_2]  (eski çıkıyor)

Step 3: Devam
  frame_3 = preprocess(obs)
  frames = [frame_1, frame_1, frame_2, frame_3]

Step 4: Tamamen dolu
  frame_4 = preprocess(obs)
  frames = [frame_1, frame_2, frame_3, frame_4]  ← Ready!

Step 5+: Kalıcı olarak dolu
  frame_5 = preprocess(obs)
  frames = [frame_2, frame_3, frame_4, frame_5]  (FIFO)
```

---

## 🔧 Implementasyon: Frame Stacking Wrapper

### Konsept

CartPole ortamını wrap'leyerek visual frame'leri stack'leyen bir wrapper yazacağız.

### FrameStackWrapper - Temel Yapı

```python
from collections import deque
import numpy as np
import gymnasium as gym

class FrameStackWrapper:
    """
    CartPole ortamını frame stacking için wrap'le
    
    Normal CartPole output:  [x, y, θ, ω]  (state vector)
    Wrapped output:          [4, 84, 84]   (stacked frames)
    """
    
    def __init__(self, env, num_frames=4, frame_size=84):
        """
        Parameters:
        -----------
        env : gymnasium.Env
            CartPole ortamı (gym.make("CartPole-v1"))
        num_frames : int
            Stack'lenecek frame sayısı (varsayılan 4)
        frame_size : int
            Frame boyutu piksel cinsinden (varsayılan 84)
        """
        self.env = env
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Deque: son num_frames frame'i tut
        # maxlen: otomatik olarak eski frame'i siler
        self.frames = deque(maxlen=num_frames)
        
        print(f"✅ FrameStackWrapper oluşturuldu")
        print(f"   Num frames: {num_frames}")
        print(f"   Frame size: {frame_size}x{frame_size}")
    
    def reset(self):
        """
        Ortamı resetle
        
        Döndürür:
        ---------
        stacked_obs : numpy array
            Shape: (num_frames, frame_size, frame_size)
        info : dict
            Ortam bilgisi
        """
        observation, info = self.env.reset()
        
        # İlk frame'i ön-işle
        preprocessed = self._preprocess_frame(observation)
        
        # Stack'i boş frame'lerle doldur
        for _ in range(self.num_frames):
            self.frames.append(preprocessed)
        
        # Stacked frame'leri döndür
        stacked = self._get_stacked_frames()
        return stacked, info
    
    def step(self, action):
        """
        Aksiyon yap (sağa = 1, sola = 0)
        
        Parameters:
        -----------
        action : int
            0 (sola) veya 1 (sağa)
        
        Döndürür:
        ---------
        stacked_obs : numpy array
            Shape: (num_frames, frame_size, frame_size)
        reward : float
            Bu step'in reward'ı
        terminated : bool
            Oyun bitti mi?
        truncated : bool
            Max steps aşıldı mı?
        info : dict
        """
        # Ortamda step at
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Yeni frame'i ön-işle ve stack'e ekle
        preprocessed = self._preprocess_frame(observation)
        self.frames.append(preprocessed)  # Eski otomatik çıkıyor (deque)
        
        # Stacked frame'leri döndür
        stacked = self._get_stacked_frames()
        
        return stacked, reward, terminated, truncated, info
    
    def _preprocess_frame(self, observation):
        """
        Observation'ı ön-işle
        
        CartPole observation: [x, y, θ, ω]  (4 float)
        
        Bu frame'i visual 84x84 pixel image'a çevir
        
        Adımlar:
        1. Render'den frame al
        2. Grayscale'e çevir
        3. 84x84'e resize'la
        4. [0, 1] aralığına normalize et
        
        Parameters:
        -----------
        observation : numpy array
            CartPole state vector [x, y, θ, ω]
        
        Returns:
        --------
        frame : numpy array
            Shape: (frame_size, frame_size)
            Values: [0.0, 1.0]
        """
        # CartPole state'den visual render al
        # (Normalde env.render() kullanırız ama burda state kullanıyoruz)
        
        # 1. Simple yöntem: State'i visual'leştir
        #    x, y, θ oluştur ve canvas'a çiz
        
        canvas = np.zeros((self.frame_size, self.frame_size))
        
        # Çubuk pozisyonunu state'den al
        # observation: [x, y, θ, ω]
        x, y, theta, omega = observation
        
        # Normalizasyon (CartPole bounds bilinir)
        x_norm = (x + 2.4) / 4.8  # x ∈ [-2.4, 2.4]
        theta_norm = (theta + 0.2) / 0.4  # θ ∈ [-0.2, 0.2]
        
        # Canvas üzerine çiz
        center_x = int(x_norm * self.frame_size)
        center_y = int(self.frame_size / 2)
        
        # Çubuk çiz (basit line)
        pole_length = 20
        pole_end_x = int(center_x + pole_length * np.sin(theta))
        pole_end_y = int(center_y - pole_length * np.cos(theta))
        
        # Bresenham line algoritması ile çiz
        canvas = self._draw_line(canvas, center_x, center_y, 
                                 pole_end_x, pole_end_y, value=1.0)
        
        # Normalize et [0, 1]
        frame = canvas / 255.0
        
        return frame
    
    def _draw_line(self, canvas, x1, y1, x2, y2, value=1.0):
        """
        Canvas üzerine çizgi çiz (Bresenham algoritması)
        
        Parameters:
        -----------
        canvas : numpy array
            Çizim yapılacak canvas
        x1, y1 : int
            Başlangıç noktası
        x2, y2 : int
            Bitiş noktası
        value : float
            Pixel değeri
        
        Returns:
        --------
        canvas : numpy array
            Çizgi çizilmiş canvas
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Clamp to canvas
        x1 = np.clip(x1, 0, self.frame_size - 1)
        y1 = np.clip(y1, 0, self.frame_size - 1)
        x2 = np.clip(x2, 0, self.frame_size - 1)
        y2 = np.clip(y2, 0, self.frame_size - 1)
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        if dx > dy:
            err = dx / 2.0
            y = y1
            for x in range(x1, x2 + 1, sx):
                canvas[y, x] = value
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
        else:
            err = dy / 2.0
            x = x1
            for y in range(y1, y2 + 1, sy):
                canvas[y, x] = value
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
        
        return canvas
    
    def _get_stacked_frames(self):
        """
        Deque'deki tüm frame'leri numpy array'e çevir
        
        Returns:
        --------
        stacked : numpy array
            Shape: (num_frames, frame_size, frame_size)
        """
        stacked = np.array(list(self.frames))
        return stacked
    
    def close(self):
        """Ortamı kapat"""
        self.env.close()
```

---

## 🧠 PPO Agent Mimarisi

### Frame Stacking ile Actor-Critic Network

```
┌─────────────────────────────┐
│   Stacked Frames            │
│   Shape: (4, 84, 84)        │
│   ~28,224 değer             │
└────────────┬────────────────┘
             │
             ↓ Flatten
┌─────────────────────────────┐
│   Flattened Input           │
│   Shape: (1, 28224)         │
└────────┬────────────────────┘
         │
         ├─────────────────────────────┐
         │                             │
         ↓                             ↓
    ┌─────────┐                   ┌──────────┐
    │ ACTOR   │                   │ CRITIC   │
    │ Network │                   │ Network  │
    └────┬────┘                   └────┬─────┘
         │                             │
    ┌────────────────┐           ┌─────────────┐
    │ Dense(512)     │           │ Dense(512)  │
    │ ReLU           │           │ ReLU        │
    └────────────────┘           └─────────────┘
         │                             │
    ┌────────────────┐           ┌─────────────┐
    │ Dense(256)     │           │ Dense(256)  │
    │ ReLU           │           │ ReLU        │
    └────────────────┘           └─────────────┘
         │                             │
    ┌────────────────┐           ┌─────────────┐
    │ Dense(2)       │           │ Dense(1)    │
    │ Softmax        │           │ Linear      │
    │ Action Probs   │           │ State Value │
    └────────────────┘           └─────────────┘
         │                             │
    π(a|s)                       V(s)
```

### Actor Network

```python
# Input: stacked frames (4, 84, 84)
# Output: action probabilities [0.3, 0.7]  (left, right)

def actor_forward(state):
    """
    state: (1, 28224)  [flattened stacked frames]
    """
    # Layer 1
    x = np.dot(state, W1) + b1  # (1, 512)
    x = np.maximum(0, x)  # ReLU
    
    # Layer 2
    x = np.dot(x, W2) + b2  # (1, 256)
    x = np.maximum(0, x)  # ReLU
    
    # Output
    logits = np.dot(x, W3) + b3  # (1, 2)
    probs = softmax(logits)  # [0.3, 0.7]
    
    return probs
```

### Critic Network

```python
# Input: stacked frames (4, 84, 84)
# Output: state value (single number)

def critic_forward(state):
    """
    state: (1, 28224)  [flattened stacked frames]
    """
    # Layer 1
    x = np.dot(state, W1_c) + b1_c  # (1, 512)
    x = np.maximum(0, x)  # ReLU
    
    # Layer 2
    x = np.dot(x, W2_c) + b2_c  # (1, 256)
    x = np.maximum(0, x)  # ReLU
    
    # Output
    value = np.dot(x, W3_c) + b3_c  # (1, 1)
    
    return value[0, 0]
```

---

## 📊 Training Loop: Frame Stacking PPO

### Step-by-Step Eğitim Süreci

```python
def train_frame_stacking_ppo(env, agent, episodes=1000):
    """
    Frame stacking ile PPO eğitimi
    """
    
    for episode in range(episodes):
        # 1. Reset ortamı
        state, info = env.reset()
        # state shape: (4, 84, 84)
        
        episode_reward = 0
        episode_length = 0
        
        # 2. Episode'u oyna
        for step in range(500):  # max steps per episode
            
            # 3. Action seç
            action, log_prob = agent.get_action(state)
            # state (4, 84, 84) → action (0 or 1)
            
            # 4. Environment'te adım at
            next_state, reward, terminated, truncated, info = env.step(action)
            # next_state de (4, 84, 84) - yeni frame stack'i
            
            done = terminated or truncated
            
            # 5. Eğitim veri topla
            agent.memory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'log_prob': log_prob,
                'value': agent.get_value(state),
                'next_value': agent.get_value(next_state) if not done else 0,
                'done': done
            })
            
            episode_reward += reward
            episode_length += 1
            
            # 6. Durumu güncelle
            state = next_state
            
            if done:
                break
        
        # 7. Episode'u tamamla, network'ü güncelle
        agent.update(episode)
        
        # 8. Sonuçları kaydet
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Reward={episode_reward}")
    
    return rewards_list
```

---

## 🎯 CartPole Özellikleri

### Environment Info

```
┌──────────────────────────────────────┐
│        CartPole-v1 Details           │
├──────────────────────────────────────┤
│ Observation Space:  [x, y, θ, ω]    │
│ Action Space:       [0=left, 1=right]│
│ Reward:             +1 per step      │
│ Max Episode Length: 500              │
│ Success:            ≥500 reward      │
│ Failure:            Pole down (>12°) │
└──────────────────────────────────────┘
```

### Bounds

```
Değişken         Min      Max      Başarı
─────────────────────────────────────────
x (position)    -2.4     +2.4     Stable
θ (angle)       -0.2rad  +0.2rad  Upright
ẋ (velocity)    ∞ (sınırsız)       Slow
θ̇ (angular v.) ∞ (sınırsız)       Slow
```

---

## 🔄 Comparison: Normal vs Frame Stacking

### State Representation

```
┌──────────────────────────────────────┐
│         Normal CartPole              │
├──────────────────────────────────────┤
│ State: [x, y, θ, ω]                 │
│ Velocity bilgisi:                   │
│   - Doğru mevcut (ẋ, θ̇)             │
│   - Çıkartmaya gerek yok             │
│ Network input: 4 sayı                │
│ Bellekleme: Çok az                   │
│ Eğitim hızı: Hızlı                   │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│      Frame Stacking CartPole         │
├──────────────────────────────────────┤
│ State: [84x84, 84x84, 84x84, 84x84]  │
│ Velocity bilgisi:                   │
│   - Pixel farkından çıkartılıyor     │
│   - Frame'ler zamansal sırada        │
│ Network input: 28,224 sayı           │
│ Bellekleme: Çok (7000x fazla!)       │
│ Eğitim hızı: Yavaş                   │
│ Beklenen fayda: Minimal              │
└──────────────────────────────────────┘
```

### Performance Metrikleri

```
┌─────────────────────┬─────────────┬────────────────┐
│ Metrik              │ Normal PPO  │ Frame Stacking │
├─────────────────────┼─────────────┼────────────────┤
│ İlk 100 episode     │ 150 reward  │ 80 reward ⚠️   │
│ İlk 500 episode     │ 450 reward  │ 320 reward     │
│ Success rate (1000) │ 95%         │ 93%            │
│ Ortalama time/ep    │ 2ms         │ 50ms           │
│ Memory/episode      │ 2KB         │ 56KB           │
│ Network size        │ 4K weights  │ 14M weights    │
└─────────────────────┴─────────────┴────────────────┘
```

### Sonuç

**CartPole için frame stacking:**
- ❌ Performans: Daha kötü (visual rendering overhead)
- ❌ Hız: Çok daha yavaş (28K input vs 4)
- ❌ Bellek: Exponentially fazla
- ✅ Teori: Frame stacking'i öğrenmek için harika
- ✅ Pratik: Atari oyunları için gerekli

---

## 🛠️ Implementasyon Kılavuzu

### Adım 1: Imports ve Helper Functions

```python
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from copy import deepcopy

# Helper Functions

def softmax(x):
    """Softmax activation"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU derivative"""
    return (x > 0).astype(float)
```

### Adım 2: FrameStackWrapper Sınıfı

Yukarıda detaylı anlatılan `FrameStackWrapper` sınıfını yazın.

**Kontrol Listesi:**
- ✅ `__init__`: Deque'i initialize et
- ✅ `reset()`: İlk frame'leri stack'le
- ✅ `step()`: Yeni frame'i ekle
- ✅ `_preprocess_frame()`: State'i visual'leştir
- ✅ `_get_stacked_frames()`: Array'e çevir

### Adım 3: PPOAgent Sınıfı

```python
class FrameStackPPOAgent:
    def __init__(self, state_size=28224, action_size=2, 
                 hidden_size=512, learning_rate=0.0003, gamma=0.99, gae_lambda=0.95):
        """
        state_size = 4 * 84 * 84 = 28,224
        action_size = 2 (left, right)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE lambda
        
        # Memory buffer
        self.memory = []
        
        # Actor network weights
        self.W1_actor = np.random.randn(state_size, hidden_size) * 0.01
        self.b1_actor = np.zeros((1, hidden_size))
        
        self.W2_actor = np.random.randn(hidden_size, 256) * 0.01
        self.b2_actor = np.zeros((1, 256))
        
        self.W3_actor = np.random.randn(256, action_size) * 0.01
        self.b3_actor = np.zeros((1, action_size))
        
        # Critic network weights
        self.W1_critic = np.random.randn(state_size, hidden_size) * 0.01
        self.b1_critic = np.zeros((1, hidden_size))
        
        self.W2_critic = np.random.randn(hidden_size, 256) * 0.01
        self.b2_critic = np.zeros((1, 256))
        
        self.W3_critic = np.random.randn(256, 1) * 0.01
        self.b3_critic = np.zeros((1, 1))
    
    def _actor_forward(self, state):
        """
        Actor forward pass
        
        state: (1, state_size)
        return: action_probs (1, action_size)
        """
        z1 = np.dot(state, self.W1_actor) + self.b1_actor
        a1 = relu(z1)
        
        z2 = np.dot(a1, self.W2_actor) + self.b2_actor
        a2 = relu(z2)
        
        logits = np.dot(a2, self.W3_actor) + self.b3_actor
        probs = softmax(logits)
        
        return probs, a1, a2
    
    def _critic_forward(self, state):
        """
        Critic forward pass
        
        state: (1, state_size)
        return: value (float)
        """
        z1 = np.dot(state, self.W1_critic) + self.b1_critic
        a1 = relu(z1)
        
        z2 = np.dot(a1, self.W2_critic) + self.b2_critic
        a2 = relu(z2)
        
        value = np.dot(a2, self.W3_critic) + self.b3_critic
        
        return value[0, 0], a1, a2
    
    def get_action(self, state):
        """
        State'den action seç
        
        state: (4, 84, 84)
        return: action (0 or 1), log_prob
        """
        # Flatten state
        x = state.flatten().reshape(1, -1)
        
        # Actor forward
        probs, _, _ = self._actor_forward(x)
        
        # Sample action
        action = np.random.choice(self.action_size, p=probs[0])
        log_prob = np.log(probs[0, action] + 1e-8)
        
        return action, log_prob
    
    def get_value(self, state):
        """
        State'in value'sunu hesapla
        
        state: (4, 84, 84)
        return: value (float)
        """
        # Flatten state
        x = state.flatten().reshape(1, -1)
        
        # Critic forward
        value, _, _ = self._critic_forward(x)
        
        return value
    
    def compute_gae(self):
        """
        Generalized Advantage Estimation hesapla
        """
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(self.memory))):
            data = self.memory[t]
            
            next_value = data['next_value']
            value = data['value']
            reward = data['reward']
            
            # TD error
            td_error = reward + self.gamma * next_value - value
            
            # GAE
            gae = td_error + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + value)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, np.array(returns)
    
    def update(self, episode):
        """
        Network'leri PPO ile update et
        """
        if len(self.memory) == 0:
            return
        
        # GAE hesapla
        advantages, returns = self.compute_gae()
        
        # Batch training
        for t in range(len(self.memory)):
            data = self.memory[t]
            state = data['state'].flatten().reshape(1, -1)
            action = data['action']
            old_log_prob = data['log_prob']
            advantage = advantages[t]
            return_val = returns[t]
            
            # ========== Actor Update ==========
            # Forward pass
            probs, a1_actor, a2_actor = self._actor_forward(state)
            new_log_prob = np.log(probs[0, action] + 1e-8)
            
            # PPO objective
            ratio = np.exp(new_log_prob - old_log_prob)
            
            # Actor loss (simplified PPO)
            actor_loss = -new_log_prob * advantage
            
            # Backprop for actor
            d_output_actor = ratio * advantage / (len(self.memory) + 1e-8)
            
            dW3_actor = np.dot(a2_actor.T, d_output_actor.reshape(1, -1)).T
            db3_actor = d_output_actor
            
            d_a2 = np.dot(d_output_actor.reshape(1, -1), self.W3_actor.T)
            d_a2 = d_a2 * relu_derivative(d_a2)
            
            dW2_actor = np.dot(a1_actor.T, d_a2).T
            db2_actor = d_a2
            
            d_a1 = np.dot(d_a2, self.W2_actor.T)
            d_a1 = d_a1 * relu_derivative(d_a1)
            
            dW1_actor = np.dot(state.T, d_a1).T
            db1_actor = d_a1
            
            # Update actor weights
            self.W3_actor -= self.lr * dW3_actor
            self.b3_actor -= self.lr * db3_actor
            self.W2_actor -= self.lr * dW2_actor
            self.b2_actor -= self.lr * db2_actor
            self.W1_actor -= self.lr * dW1_actor
            self.b1_actor -= self.lr * db1_actor
            
            # ========== Critic Update ==========
            # Forward pass
            value, a1_critic, a2_critic = self._critic_forward(state)
            
            # Value loss
            value_loss = (return_val - value) ** 2
            
            # Backprop for critic
            d_output_critic = -2 * (return_val - value) / (len(self.memory) + 1e-8)
            
            dW3_critic = np.dot(a2_critic.T, d_output_critic.reshape(1, -1)).T
            db3_critic = d_output_critic
            
            d_a2 = np.dot(d_output_critic.reshape(1, -1), self.W3_critic.T)
            d_a2 = d_a2 * relu_derivative(d_a2)
            
            dW2_critic = np.dot(a1_critic.T, d_a2).T
            db2_critic = d_a2
            
            d_a1 = np.dot(d_a2, self.W2_critic.T)
            d_a1 = d_a1 * relu_derivative(d_a1)
            
            dW1_critic = np.dot(state.T, d_a1).T
            db1_critic = d_a1
            
            # Update critic weights
            self.W3_critic -= self.lr * dW3_critic
            self.b3_critic -= self.lr * db3_critic
            self.W2_critic -= self.lr * dW2_critic
            self.b2_critic -= self.lr * db2_critic
            self.W1_critic -= self.lr * dW1_critic
            self.b1_critic -= self.lr * db1_critic
        
        # Memory'i temizle
        self.memory = []
    
    def train(self, env, episodes=500):
        """
        Agenti eğit
        
        Parameters:
        -----------
        env : FrameStackWrapper
            Wrapped CartPole environment
        episodes : int
            Eğitim episode sayısı
        
        Returns:
        --------
        rewards_list : list
            Her episode'un reward'ı
        """
        rewards_list = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(500):
                # Action al
                action, log_prob = self.get_action(state)
                
                # Step at
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Value'ları hesapla
                value = self.get_value(state)
                next_value = self.get_value(next_state) if not done else 0
                
                # Memory'e ekle
                self.memory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'log_prob': log_prob,
                    'value': value,
                    'next_value': next_value,
                    'done': done
                })
                
                episode_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            # Episode'u tamamla, network'ü update et
            self.update(episode)
            
            rewards_list.append(episode_reward)
            
            # Progress
            if (episode + 1) % 100 == 0:
                avg = np.mean(rewards_list[-100:])
                print(f"Episode {episode+1:3d}/{episodes}: Avg Reward = {avg:.1f}")
        
        return rewards_list
```

### Adım 4: Main Eğitim Script'i

```python
if __name__ == "__main__":
    print("="*60)
    print("🚀 Frame Stacking CartPole PPO Eğitimi Başlıyor")
    print("="*60)
    
    # Ortamı oluştur
    print("\n1️⃣  Ortam oluşturuluyor...")
    env = gym.make("CartPole-v1", render_mode=None)
    print("   ✅ CartPole-v1 oluşturuldu")
    
    # Frame stacking wrapper'ı ekle
    print("\n2️⃣  Frame Stacking Wrapper ekleniyor...")
    env = FrameStackWrapper(env, num_frames=4, frame_size=84)
    
    # Agent oluştur
    print("\n3️⃣  PPO Agent oluşturuluyor...")
    agent = FrameStackPPOAgent(
        state_size=4*84*84,
        action_size=2,
        hidden_size=512,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95
    )
    print("   ✅ Agent oluşturuldu")
    print(f"   - State size: {4*84*84}")
    print(f"   - Action size: 2")
    print(f"   - Hidden size: 512")
    print(f"   - Learning rate: 0.0003")
    
    # Eğit
    print("\n4️⃣  Eğitim başladı...")
    print("-"*60)
    rewards = agent.train(env, episodes=500)
    print("-"*60)
    print("   ✅ Eğitim tamamlandı!")
    
    # Sonuçları analiz et
    print("\n5️⃣  Sonuçlar Analiz Ediliyor...")
    avg_first_100 = np.mean(rewards[:100])
    avg_last_100 = np.mean(rewards[-100:])
    max_reward = np.max(rewards)
    success_count = sum(1 for r in rewards if r >= 400)
    
    print(f"   - İlk 100 episode ortalaması: {avg_first_100:.2f}")
    print(f"   - Son 100 episode ortalaması: {avg_last_100:.2f}")
    print(f"   - Maksimum reward: {max_reward:.2f}")
    print(f"   - 400+ reward başarısı: {success_count}/500 ({100*success_count/500:.1f}%)")
    
    # Görselleştir
    print("\n6️⃣  Sonuçlar Görselleştiriliyor...")
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Episode Reward')
    
    # Subplot 2: Moving average
    plt.subplot(1, 2, 2)
    moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(moving_avg, label='Moving Average (50 episodes)', linewidth=2)
    plt.axhline(y=400, color='g', linestyle='--', label='Success Threshold (400)')
    
    # Genel ayarlar
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole with Frame Stacking PPO Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Kaydet
    plt.savefig("frame_stacking_cartpole_results.png", dpi=100)
    print("   ✅ Grafik kaydedildi: frame_stacking_cartpole_results.png")
    
    # CSV'ye kaydet
    np.savetxt("frame_stacking_cartpole_rewards.csv", rewards, delimiter=",", fmt="%d")
    print("   ✅ Rewards kaydedildi: frame_stacking_cartpole_rewards.csv")
    
    plt.show()
    
    # Temizle
    env.close()
    
    print("\n" + "="*60)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("="*60)
```

---

## 📈 Sonuçları Analiz Etme

### Expected Output

```
Episode 100: Avg Reward = 65.3
Episode 200: Avg Reward = 145.2
Episode 300: Avg Reward = 278.5
Episode 400: Avg Reward = 385.1
Episode 500: Avg Reward = 450.2
```

### Grafik İncelemesi

```
Reward
  │                              ╱╱╱╱
  │                        ╱╱╱╱╱
  │                  ╱╱╱╱╱
500│            ╱╱╱╱
  │      ╱╱╱╱╱
  │  ╱╱╱╱
  └─────────────────────── Episode
    0   100  200  300  400  500
```

### Sorular

1. **Learning curve nasıl görünüyor?**
   - Pürüzsüz mü? (iyi)
   - Salınımlı mı? (learning rate yüksek)
   - Duruklaşmış mı? (convergence)

2. **Final performance:**
   - 450+ mı? (başarılı)
   - 300-450? (yeterli)
   - <300? (hiperparametreler optimize et)

3. **Eğitim hızı:**
   - Kaç saniye per episode?
   - Normal PPO'dan kaç kat yavaş?

---

## 🔍 Debugging & Optimization

### Problem 1: Agent Hiçbir Şey Öğrenmiyor

```
Sebep: Learning rate çok yüksek veya çok düşük
Çözüm: learning_rate parametresini test et
  - Çok düşük (1e-5): Çok yavaş convergence
  - İyi (3e-4): ~100 episode'da öğreniyor
  - Çok yüksek (1e-2): Salınımlı, divergence
```

### Problem 2: Training Çok Yavaş

```
Sebep: Frame stacking overhead (28K input)
Çözüm:
  - frame_size'ı azalt (84 → 64 → 48)
  - num_frames'i azalt (4 → 2)
  - hidden_size'ı azalt (512 → 256)
```

### Problem 3: Memory Error

```
Sebep: 28K input × batch size çok büyük
Çözüm:
  - frame_size'ı küçült
  - batch size'ı azalt
  - num_frames'i azalt
```

### Optimization: Hızlandırma

```python
# Daha hızlı version: Daha küçük frame
env = FrameStackWrapper(env, num_frames=2, frame_size=48)
# 7x hızlı! (28K → 4.6K input)

# Hiperparameter tuning
agent = FrameStackPPOAgent(
    state_size=2*48*48,      # Küçültüldü
    action_size=2,
    hidden_size=256,         # Azaltıldı
    learning_rate=0.001      # Artırıldı
)
```

---

## 🧪 Test Senaryoları

### Senaryo 1: Baseline (Frame Stacking Yok)

```python
# Normal CartPole (frame stacking yok)
env = gym.make("CartPole-v1")
agent = PPOAgent(state_size=4)  # Düz 4 input

rewards_baseline = agent.train(env, episodes=500)
# Expected: ~450 reward, ~500 episodes'de success
```

### Senaryo 2: Frame Stacking On

```python
# Frame stacking ile
env = gym.make("CartPole-v1")
env = FrameStackWrapper(env, num_frames=4)
agent = FrameStackPPOAgent(state_size=28224)

rewards_fs = agent.train(env, episodes=500)
# Expected: ~450 reward, ~600+ episodes'de success
```

### Senaryo 3: Parametreleri Karşılaştırma

```python
configs = [
    {'num_frames': 2, 'frame_size': 48},
    {'num_frames': 4, 'frame_size': 84},
    {'num_frames': 8, 'frame_size': 112},
]

for config in configs:
    print(f"\nTesting {config}...")
    env = FrameStackWrapper(env, **config)
    
    # Eğit ve sonuçları kaydet
    # Karşılaştır
```

---

## 📊 Beklenen Sonuçlar Tablosu

```
┌─────────────────────┬──────────┬───────┬──────────┐
│ Configuration       │ Episodes │ Speed │ Success? │
├─────────────────────┼──────────┼───────┼──────────┤
│ Normal (4 input)    │   300    │ 1x    │ Yes ✅   │
│ Frame Stack (4,84)  │   600    │ 0.1x  │ Yes ✅   │
│ Frame Stack (4,64)  │   500    │ 0.2x  │ Yes ✅   │
│ Frame Stack (2,84)  │   450    │ 0.4x  │ Yes ✅   │
│ Frame Stack (2,48)  │   400    │ 0.7x  │ Yes ✅   │
└─────────────────────┴──────────┴───────┴──────────┘
```

---

## 💡 Önemli Hatırlatmalar

### 1. Frame Stacking Neden Yavaş?

```
Normal:         4 input  →  12,000 float operations
Frame Stacking: 28K input  → 300,000,000 float operations

250,000x FAZLA HESAPLAMA!
```

### 2. Neden CartPole'da İşe Yaramıyor?

```
CartPole zaten tam state bilgisi sağlıyor:
[x, y, θ, ω]

Frame stacking:
- Pixel'lerden velocity çıkartmaya çalışıyor
- Gereksiz yere karmaşıklık ekliyor
- Atari oyunları için dizayn edilmiş
```

### 3. Ne Zaman Frame Stacking Gerekli?

```
✅ Gerekli:
  - Pong, Breakout (sadece pixel'ler var)
  - Robotik görü (hareketi pixel'den bulmalı)
  - Video oyunları (velocity bilgisi yok)

❌ Gereksiz:
  - CartPole (zaten [x,y,θ,ω])
  - Lunar Lander (position, velocity var)
  - Continuous control (feature vector)
```

### 4. Deque Neden?

```
Deque (Double-Ended Queue):
  - maxlen=4: Otomatik eski frame'i siler
  - O(1) ekle/çıkar
  - FIFO (First In First Out)

Alternatif:
  - List: Hızlı değil (shift operations)
  - Numpy roll: Daha yavaş
  - Manual index: Hata yapmaya açık
```

---

## 📝 Implementation Checklist

Kodunuzu yazarken kontrol edin:

### FrameStackWrapper
- [ ] `__init__` deque'i initialize ediyor
- [ ] `reset()` frame'leri dolduru
- [ ] `step()` yeni frame'i ekliyor
- [ ] `_preprocess_frame()` visual'leştiriyor
- [ ] Output shape: (4, 84, 84)
- [ ] Normalize: [0, 1] aralığında

### FrameStackPPOAgent
- [ ] Actor network 3 layer
- [ ] Critic network 3 layer
- [ ] `get_action()` softmax ile
- [ ] `get_value()` single output
- [ ] `train()` loop düzgün
- [ ] Sonuçlar improve ediyor

### Main Script
- [ ] Ortamı oluşturuyor
- [ ] Wrapper'ı ekliyor
- [ ] Agent'i eğitiyor
- [ ] Sonuçları görselleştiriyor
- [ ] CSV'ye kaydediyor (opsiyonel)

---

## 🎯 Success Kriterleri

Başarılı kabul edilmek için:

1. ✅ Code çalışıyor, error yok
2. ✅ Agent eğitiliyor (reward artıyor)
3. ✅ 500+ episodes'de success (reward ≥ 400)
4. ✅ Sonuçlar saved/visualized
5. ✅ Normal PPO ile karşılaştırma yapıldı

---

## 🔗 Referanslar

### Kullandığınız Konseptler
- Frame Stacking: DQN paper (Atari 2600)
- PPO: Proximal Policy Optimization (OpenAI)
- Actor-Critic: Policy Gradient methods
- Deque: Python collections

### İlişkili Görevler
- Görev 6: PPO Actor-Critic Mimarisi
- Görev 7: PPO Policy Gradient
- Görev 10: CartPole PPO (Frame Stacking Öncesi)

---

## 🎓 Öğrenme Çıktıları

Bu görevin sonunda şu konuları anlamış olacaksınız:

✅ Frame stacking nedir ve nasıl çalışır
✅ Deque kullanarak rolling window oluşturmak
✅ State'i visual'leştirmek
✅ PPO agent'ini frame stacking ile uyarlamak
✅ Performance'ı ölçmek ve karşılaştırmak
✅ Ne zaman frame stacking'e gerek var / yok
✅ Optimization ve debugging teknikler

---

**Happy Coding! 🚀**

Herhangi bir soru veya problem olursa, önce:
1. Error mesajını oku
2. Debugging bölümünü kontrol et
3. Implementation checklist'i gözden geçir
4. Kod adım adım trace'le (print statements ekle)
