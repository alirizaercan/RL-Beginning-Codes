# 🎯 Görev 5: CartPole ile DQN - Hepsi Bir Arada!

## 📚 Teorik Bilgi

### Deep Q-Network (DQN) Algoritması

Artık tüm parçaları öğrendin! Şimdi hepsini birleştirip **gerçek bir RL problemi** çözeceğiz!

**DQN Bileşenleri:**
1. ✅ Q-Network (Forward + Backward)
2. ✅ Experience Replay Buffer
3. ✅ Epsilon-Greedy Exploration
4. 🆕 Target Network (stability için)
5. 🆕 Training Loop

### CartPole Problemi

**Görev:** Bir çubuğu dik tutmak (çubuğu dengelemek)

**State (4 boyut):**
- Position: Arabanın pozisyonu (-4.8 ile 4.8 arası)
- Velocity: Arabanın hızı
- Angle: Çubuğun açısı (radyan)
- Angular Velocity: Çubuğun açısal hızı

**Actions (2 boyut):**
- 0: Sola git
- 1: Sağa git

**Reward:**
- Her adımda +1 (çubuk dik kaldığı sürece)

**Episode Sonu:**
- Çubuk 15 dereceden fazla eğilirse
- Araba ekrandan çıkarsa
- 500 adım geçerse (başarı!)

**Hedef:** Ortalama 195+ reward (100 episode üzerinden)

---

## 🛠️ Adım 1: DQN Agent Class'ı

Yeni dosya: `cartpole_dqn.py`

```python
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from q_learning_nn import QNetwork
from experience_replay import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network Agent
    
    Q-Learning + Neural Network + Experience Replay
    """
    
    def __init__(
        self, 
        state_size, 
        action_size, 
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64
    ):
        """
        DQN Agent'ı başlat
        
        Parameters:
        -----------
        state_size : int
            State boyutu
        action_size : int
            Action sayısı
        gamma : float
            Discount factor (gelecek reward'ların önemi)
        epsilon_* : float
            Exploration parametreleri
        buffer_capacity : int
            Replay buffer kapasitesi
        batch_size : int
            Her update'te kaç deneyim kullanılacak
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        
        # TODO: Epsilon (exploration) parametreleri
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # TODO: Q-Network oluştur
        self.q_network = QNetwork(
            state_size, 
            action_size, 
            hidden_size, 
            learning_rate
        )
        
        # TODO: Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        print(f"✅ DQN Agent oluşturuldu!")
        print(f"   State size: {state_size}")
        print(f"   Action size: {action_size}")
        print(f"   Gamma: {gamma}")
        print(f"   Batch size: {batch_size}")
    
    def select_action(self, state):
        """
        Epsilon-greedy ile action seç
        """
        return self.q_network.get_action(state, epsilon=self.epsilon)
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Deneyimi buffer'a ekle
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        """
        Replay buffer'dan batch çekip network'ü güncelle
        
        Returns:
        --------
        loss : float
            Training loss (None if not enough samples)
        """
        # TODO: Yeterli deneyim var mı?
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # TODO: Batch sample
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # TODO: Current Q-values
        current_q_values = self.q_network.predict(states)
        
        # TODO: Next Q-values (target için)
        next_q_values = self.q_network.predict(next_states)
        
        # TODO: Target Q-values hesapla (Bellman Equation)
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                # Episode bittiyse sadece reward
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # Bellman equation: Q(s,a) = r + γ * max Q(s',a')
                target_q_values[i, actions[i]] = \
                    rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # TODO: Backward pass (network'ü güncelle)
        self.q_network.backward(target_q_values)
        
        # TODO: Loss hesapla
        loss = np.mean((current_q_values - target_q_values) ** 2)
        
        return loss
    
    def decay_epsilon(self):
        """
        Epsilon'u azalt (exploration → exploitation)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

### 🔍 Kod Açıklaması

**Bellman Equation:**
```python
if done:
    target = reward  # Episode bitti, gelecek yok
else:
    target = reward + gamma * max(Q(next_state))  # Gelecek reward'ları da ekle
```

**Epsilon Decay:**
```python
# Her episode sonra
epsilon = max(0.01, epsilon * 0.995)
# 1.0 → 0.995 → 0.990 → ... → 0.01
```

---

## 🛠️ Adım 2: Training Loop

```python
def train_dqn(episodes=500, render=False):
    """
    DQN ile CartPole eğit
    
    Parameters:
    -----------
    episodes : int
        Kaç episode eğitim yapılacak
    render : bool
        Ortamı görselleştir mi?
    """
    # TODO: Ortam oluştur
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # TODO: Agent oluştur
    agent = DQNAgent(
        state_size=4,
        action_size=2,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64
    )
    
    # TODO: Metrikleri sakla
    episode_rewards = []
    episode_losses = []
    
    print("\n" + "="*60)
    print("🚀 Eğitim Başlıyor!")
    print("="*60)
    
    for episode in range(episodes):
        # Episode başlat
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # TODO: Action seç
            action = agent.select_action(state)
            
            # TODO: Environment'ta action'ı uygula
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # TODO: Deneyimi sakla
            agent.store_transition(state, action, reward, next_state, done)
            
            # TODO: Train et (eğer yeterli deneyim varsa)
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
        
        # TODO: Epsilon decay
        agent.decay_epsilon()
        
        # TODO: Metrikleri kaydet
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # TODO: İlerleme göster
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1:3d} | "
                  f"Reward: {episode_reward:3.0f} | "
                  f"Avg(10): {avg_reward:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # TODO: Hedef başarı kontrolü
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            if avg_100 >= 195:
                print(f"\n🎉 Çözüldü! Episode {episode+1}'de")
                print(f"   Son 100 episode ortalaması: {avg_100:.1f}")
                break
    
    env.close()
    
    return episode_rewards, episode_losses, agent
```

---

## 🛠️ Adım 3: Görselleştirme ve Test

```python
def plot_results(rewards, losses):
    """
    Eğitim sonuçlarını görselleştir
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Reward grafiği
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')
    
    # Moving average
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                 'r-', linewidth=2, label=f'Moving Avg ({window})')
    
    ax1.axhline(y=195, color='g', linestyle='--', label='Hedef (195)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Eğitim İlerlemesi - Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Loss grafiği
    ax2.plot(losses, alpha=0.6, label='Episode Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Eğitim İlerlemesi - Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('neural-network/cartpole_dqn_results.png')
    print(f"\n📊 Grafik kaydedildi: neural-network/cartpole_dqn_results.png")
    plt.show()

def test_agent(agent, episodes=10, render=True):
    """
    Eğitilmiş agent'ı test et (epsilon=0, sadece exploit)
    """
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    print("\n" + "="*60)
    print("🧪 Test Başlıyor! (Epsilon = 0, Sadece Exploit)")
    print("="*60)
    
    test_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon = 0: Sadece en iyi action
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}: Reward = {episode_reward}")
    
    env.close()
    
    avg_reward = np.mean(test_rewards)
    print(f"\n✅ Ortalama Test Reward: {avg_reward:.1f}")
    
    return test_rewards
```

---

## 🛠️ Adım 4: Main Program

```python
if __name__ == "__main__":
    # Eğitim
    print("🎮 CartPole DQN - Tam Eğitim")
    
    rewards, losses, agent = train_dqn(episodes=500, render=False)
    
    # Sonuçları görselleştir
    plot_results(rewards, losses)
    
    # İstatistikler
    print("\n" + "="*60)
    print("📊 Eğitim İstatistikleri")
    print("="*60)
    print(f"Toplam episode: {len(rewards)}")
    print(f"En yüksek reward: {np.max(rewards):.0f}")
    print(f"Son 10 episode ort: {np.mean(rewards[-10:]):.1f}")
    print(f"Son 100 episode ort: {np.mean(rewards[-100:]):.1f}")
    
    # Test et (görsel olarak)
    input("\n▶️ Test için Enter'a bas...")
    test_rewards = test_agent(agent, episodes=5, render=True)
```

---

## 🎯 Beklenen Çıktı

```
============================================================
🚀 Eğitim Başlıyor!
============================================================
✅ DQN Agent oluşturuldu!
   State size: 4
   Action size: 2
   Gamma: 0.99
   Batch size: 64

Episode  10 | Reward:  18 | Avg(10): 20.3 | Loss: 0.0234 | ε: 0.951
Episode  20 | Reward:  25 | Avg(10): 23.1 | Loss: 0.0456 | ε: 0.905
Episode  30 | Reward:  45 | Avg(10): 35.4 | Loss: 0.0389 | ε: 0.861
Episode  40 | Reward:  52 | Avg(10): 48.7 | Loss: 0.0312 | ε: 0.819
...
Episode 150 | Reward: 189 | Avg(10): 175.3 | Loss: 0.0089 | ε: 0.256
Episode 160 | Reward: 210 | Avg(10): 195.8 | Loss: 0.0067 | ε: 0.243

🎉 Çözüldü! Episode 162'de
   Son 100 episode ortalaması: 196.3

============================================================
📊 Eğitim İstatistikleri
============================================================
Toplam episode: 162
En yüksek reward: 500
Son 10 episode ort: 195.8
Son 100 episode ort: 196.3

📊 Grafik kaydedildi: neural-network/cartpole_dqn_results.png

▶️ Test için Enter'a bas...

============================================================
🧪 Test Başlıyor! (Epsilon = 0, Sadece Exploit)
============================================================
Test Episode 1: Reward = 500.0
Test Episode 2: Reward = 500.0
Test Episode 3: Reward = 500.0
Test Episode 4: Reward = 500.0
Test Episode 5: Reward = 500.0

✅ Ortalama Test Reward: 500.0
```

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `DQNAgent` class'ını oluşturdum
- [ ] `select_action()`: Epsilon-greedy
- [ ] `store_transition()`: Buffer'a ekleme
- [ ] `train()`: Bellman equation ile target hesaplama
- [ ] `decay_epsilon()`: Exploration azaltma
- [ ] `train_dqn()`: Ana training loop
- [ ] `plot_results()`: Görselleştirme
- [ ] `test_agent()`: Test fonksiyonu
- [ ] Eğitimi çalıştırdım (195+ reward)
- [ ] Test ettim ve başarılı oldu

---

## 🎓 Ne Öğrendim?

1. **DQN Pipeline**: Ortam → Agent → Buffer → Train → Update
2. **Bellman Equation**: Future reward'ları discount ile ekle
3. **Epsilon Decay**: Explore'dan exploit'e geçiş
4. **Training Loop**: Episode → Step → Store → Train
5. **Convergence**: Loss azalıyor, reward artıyor → başarı!

---

## 🧠 Derinlemesine Sorular

1. **Gamma = 0.5 olsaydı ne değişirdi?** (kısa görüşlü ajan)
2. **Batch size = 1 olsaydı?** (online learning)
3. **Hidden size = 256 daha mı iyi?** (overfit riski)
4. **Target Network neden yok?** (DQN paper'da var)
5. **Double DQN ne ekler?** (overestimation bias)

---

## 🎉 Tebrikler!

Q-Learning için Neural Network'ü **sıfırdan** yazdın ve **gerçek bir RL problemi** çözdün!

### Öğrendiklerin:
✅ Neural Network (Forward + Backward)  
✅ Backpropagation & Gradient Descent  
✅ Experience Replay Buffer  
✅ Epsilon-Greedy Exploration  
✅ Q-Learning & Bellman Equation  
✅ DQN Training Pipeline  

---

## 🚀 Sonraki Adım: PPO

Şimdi **Policy Gradient** metodlarına geçiyoruz! PPO (Proximal Policy Optimization) daha güçlü ve modern bir algoritma.

**Görev 6**: PPO için Actor-Critic NN mimarisi

Hazır olduğunda devam edelim! 🎯
