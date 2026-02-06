# 🎯 Görev 10: CartPole ile PPO - Tam Implementation!

## 📚 Teorik Bilgi

### PPO vs DQN Karşılaştırması

| Özellik | DQN | PPO |
|---------|-----|-----|
| **Yaklaşım** | Value-based | Policy-based |
| **Output** | Q-values | Action probabilities |
| **Action** | Deterministik (argmax) | Stochastic (sample) |
| **Data** | Replay buffer | On-policy trajectory |
| **Öğrenme** | Off-policy | On-policy |
| **Continuous** | ❌ Zor | ✅ Kolay |
| **Stability** | Replay + Target net | Clip + Multiple epochs |

### PPO'nun Avantajları

1. **🎯 Stochastic Policy**: Exploration doğal
2. **🚀 Stable**: Clip mekanizması
3. **📊 Data Efficient**: Multiple epochs
4. **🎮 Versatile**: Discrete & continuous
5. **🏆 State-of-the-art**: Modern RL'de çok kullanılır

---

## 🛠️ Adım 1: Training Loop

Yeni dosya: `cartpole_ppo.py`

```python
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent

def train_ppo(
    episodes=300,
    max_steps=500,
    trajectory_length=2048,
    render=False
):
    """
    PPO ile CartPole eğit
    
    Parameters:
    -----------
    episodes : int
        Maksimum episode sayısı
    max_steps : int
        Episode başına max adım
    trajectory_length : int
        Kaç step topla, sonra train et
    render : bool
        Görselleştir mi?
    
    Returns:
    --------
    episode_rewards : list
        Her episode'un reward'ı
    agent : PPOAgent
        Eğitilmiş agent
    """
    # TODO: Environment
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # TODO: PPO Agent
    agent = PPOAgent(
        state_size=4,
        action_size=2,
        hidden_sizes=[128, 64],
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=64
    )
    
    # Metrics
    episode_rewards = []
    all_actor_losses = []
    all_critic_losses = []
    all_clip_fractions = []
    
    print("\n" + "="*60)
    print("🚀 PPO Training Başlıyor!")
    print("="*60)
    
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    steps = 0
    
    for step in range(trajectory_length * episodes):
        # TODO: Action seç
        action, log_prob, value = agent.select_action(state)
        
        # TODO: Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # TODO: Store transition
        agent.store_transition(state, action, reward, done, log_prob, value)
        
        episode_reward += reward
        state = next_state
        steps += 1
        
        # Episode bitti mi?
        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)
            
            # İlerleme göster
            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode_count:3d} | "
                      f"Reward: {episode_reward:3.0f} | "
                      f"Avg(10): {avg_reward:.1f} | "
                      f"Steps: {steps}")
            
            # Reset
            state, _ = env.reset()
            episode_reward = 0
            
            # Başarı kontrolü
            if len(episode_rewards) >= 100:
                avg_100 = np.mean(episode_rewards[-100:])
                if avg_100 >= 195:
                    print(f"\n🎉 Çözüldü! Episode {episode_count}'de")
                    print(f"   Son 100 episode ortalaması: {avg_100:.1f}")
                    break
        
        # TODO: Train et (trajectory doldu mu?)
        if len(agent.buffer) >= trajectory_length:
            metrics = agent.train()
            
            if metrics:
                all_actor_losses.append(metrics['actor_loss'])
                all_critic_losses.append(metrics['critic_loss'])
                all_clip_fractions.append(metrics['clip_fraction'])
                
                # Training info
                print(f"   📊 Training: Actor Loss={metrics['actor_loss']:.4f}, "
                      f"Critic Loss={metrics['critic_loss']:.4f}, "
                      f"Clip={metrics['clip_fraction']:.2%}")
            
            # Buffer temizlendi, yeni trajectory'ye başla
            steps = 0
        
        # Max episode kontrolü
        if episode_count >= episodes:
            break
    
    env.close()
    
    return episode_rewards, all_actor_losses, all_critic_losses, all_clip_fractions, agent
```

---

## 🛠️ Adım 2: Görselleştirme

```python
def plot_results(rewards, actor_losses, critic_losses, clip_fractions):
    """
    Training sonuçlarını görselleştir
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward plot
    ax1 = axes[0, 0]
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
    
    # Actor loss
    ax2 = axes[0, 1]
    if len(actor_losses) > 0:
        ax2.plot(actor_losses, alpha=0.6, label='Actor Loss')
        ax2.set_xlabel('Training Update')
        ax2.set_ylabel('Loss')
        ax2.set_title('Actor Loss')
        ax2.legend()
        ax2.grid(True)
    
    # Critic loss
    ax3 = axes[1, 0]
    if len(critic_losses) > 0:
        ax3.plot(critic_losses, alpha=0.6, label='Critic Loss', color='orange')
        ax3.set_xlabel('Training Update')
        ax3.set_ylabel('Loss')
        ax3.set_title('Critic Loss')
        ax3.legend()
        ax3.grid(True)
    
    # Clip fraction
    ax4 = axes[1, 1]
    if len(clip_fractions) > 0:
        ax4.plot(clip_fractions, alpha=0.6, label='Clip Fraction', color='purple')
        ax4.axhline(y=0.1, color='r', linestyle='--', label='Healthy (<10%)')
        ax4.set_xlabel('Training Update')
        ax4.set_ylabel('Clip Fraction')
        ax4.set_title('PPO Clip Fraction')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('neural-network/cartpole_ppo_results.png')
    print(f"\n📊 Grafik kaydedildi: neural-network/cartpole_ppo_results.png")
    plt.show()

def test_agent(agent, episodes=10, render=True):
    """
    Eğitilmiş PPO agent'ı test et
    """
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    print("\n" + "="*60)
    print("🧪 Test Başlıyor! (Stochastic Policy)")
    print("="*60)
    
    test_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Stochastic action (sampling)
            action, _, _ = agent.select_action(state)
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

## 🛠️ Adım 3: Main Program

```python
if __name__ == "__main__":
    print("🎮 CartPole PPO - Tam Eğitim")
    
    # Eğitim
    rewards, actor_losses, critic_losses, clip_fracs, agent = train_ppo(
        episodes=300,
        trajectory_length=2048,
        render=False
    )
    
    # Sonuçları görselleştir
    plot_results(rewards, actor_losses, critic_losses, clip_fracs)
    
    # İstatistikler
    print("\n" + "="*60)
    print("📊 Eğitim İstatistikleri")
    print("="*60)
    print(f"Toplam episode: {len(rewards)}")
    print(f"En yüksek reward: {np.max(rewards):.0f}")
    print(f"Son 10 episode ort: {np.mean(rewards[-10:]):.1f}")
    if len(rewards) >= 100:
        print(f"Son 100 episode ort: {np.mean(rewards[-100:]):.1f}")
    
    # Test et (görsel olarak)
    input("\n▶️ Test için Enter'a bas...")
    test_rewards = test_agent(agent, episodes=5, render=True)
```

---

## 🎯 Beklenen Çıktı

```
🎮 CartPole PPO - Tam Eğitim
✅ Actor Network oluşturuldu:
   4 → 128 → 64 → 2
✅ Critic Network oluşturuldu:
   4 → 128 → 64 → 1
✅ PPO Agent oluşturuldu!
   Epochs: 10, Mini-batch: 64
   Clip ε: 0.2, γ: 0.99, λ: 0.95

============================================================
🚀 PPO Training Başlıyor!
============================================================
Episode  10 | Reward:  23 | Avg(10): 21.3 | Steps: 213
   📊 Training: Actor Loss=0.0123, Critic Loss=0.4567, Clip=5.23%
Episode  20 | Reward:  34 | Avg(10): 28.7 | Steps: 287
   📊 Training: Actor Loss=0.0089, Critic Loss=0.3245, Clip=4.12%
Episode  30 | Reward:  56 | Avg(10): 43.2 | Steps: 432
   📊 Training: Actor Loss=0.0067, Critic Loss=0.2134, Clip=3.45%
...
Episode 100 | Reward: 178 | Avg(10): 145.6 | Steps: 1456
   📊 Training: Actor Loss=0.0034, Critic Loss=0.0789, Clip=2.67%
Episode 110 | Reward: 234 | Avg(10): 189.3 | Steps: 1893
   📊 Training: Actor Loss=0.0028, Critic Loss=0.0456, Clip=2.12%

🎉 Çözüldü! Episode 115'de
   Son 100 episode ortalaması: 196.8

📊 Grafik kaydedildi: neural-network/cartpole_ppo_results.png

============================================================
📊 Eğitim İstatistikleri
============================================================
Toplam episode: 115
En yüksek reward: 500
Son 10 episode ort: 218.4
Son 100 episode ort: 196.8

▶️ Test için Enter'a bas...

============================================================
🧪 Test Başlıyor! (Stochastic Policy)
============================================================
Test Episode 1: Reward = 500.0
Test Episode 2: Reward = 500.0
Test Episode 3: Reward = 500.0
Test Episode 4: Reward = 456.0
Test Episode 5: Reward = 500.0

✅ Ortalama Test Reward: 491.2
```

---

## ✅ Kontrol Listesi

Bu görevi tamamlamak için:

- [ ] `train_ppo()`: Ana training loop
- [ ] Trajectory collection (2048 steps)
- [ ] Multiple episodes tek trajectory'de
- [ ] Training metrics monitoring
- [ ] `plot_results()`: 4 grafik (reward, actor, critic, clip)
- [ ] `test_agent()`: Test fonksiyonu
- [ ] Eğitimi çalıştırdım (195+ reward)
- [ ] Test ettim ve başarılı oldu
- [ ] Clip fraction < 10% (healthy)

---

## 🎓 Ne Öğrendim?

1. **PPO Pipeline**: Collect → Compute Advantages → Train Multiple Epochs
2. **Trajectory-based**: On-policy learning
3. **Stochastic Policy**: Natural exploration
4. **Clip Monitoring**: Healthy < 10%
5. **Sample Efficiency**: DQN'den daha hızlı öğrenir

---

## 📊 PPO vs DQN Sonuçları (CartPole)

| Metrik | DQN | PPO |
|--------|-----|-----|
| **Başarı Episode** | 150-200 | 100-150 |
| **Stability** | Orta (loss patlayabilir) | Yüksek (clip sayesinde) |
| **Data Efficiency** | Düşük (replay buffer) | Yüksek (multiple epochs) |
| **Exploration** | Epsilon-greedy | Stochastic policy |
| **Test Reward** | 500 (deterministik) | 490+ (stochastic) |

---

## 🧠 Derinlemesine Sorular

1. **Trajectory 512 olsaydı?** (daha sık update)
2. **Epochs = 20 olsaydı?** (overfitting riski)
3. **Learning rate 0.001 olsaydı?** (daha yavaş)
4. **Clip epsilon = 0.1 olsaydı?** (daha conservative)

---

## 🏆 Bonus: İyileştirmeler

### 1. Learning Rate Scheduling
```python
def lr_schedule(initial_lr, step, total_steps):
    return initial_lr * (1 - step / total_steps)
```

### 2. Entropy Bonus
```python
# Exploration için entropy ekle
entropy = -np.sum(action_probs * np.log(action_probs + 1e-10), axis=1)
actor_loss -= 0.01 * np.mean(entropy)  # Entropy coefficient
```

### 3. Value Clipping
```python
# Value'da da clip uygula (stability)
v_clipped = old_values + np.clip(values - old_values, -clip_epsilon, clip_epsilon)
value_loss = np.maximum((values - returns)**2, (v_clipped - returns)**2)
```

### 4. KL Divergence Monitoring
```python
# Policy ne kadar değişti?
kl = np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1)
mean_kl = np.mean(kl)

# Eğer KL çok büyükse early stopping
if mean_kl > 0.015:  # KL threshold
    break  # Bu epoch'u durdur
```

---

## 🎉 Tebrikler!

### 🏅 Başarılarınız:

✅ **Q-Learning DQN** - Value-based RL  
✅ **PPO** - Policy-based RL  
✅ **Actor-Critic** - İki network birlikte  
✅ **Advantage Estimation** - GAE  
✅ **PPO Clip** - Stability mechanism  
✅ **CartPole Çözümü** - İki farklı algoritma ile!  

### 📚 Öğrendikleriniz:

1. Neural Network (sıfırdan!)
2. Backpropagation & Gradient Descent
3. Q-Learning & Bellman Equation
4. Policy Gradient Theorem
5. Actor-Critic Architecture
6. PPO Clipped Objective
7. GAE (Generalized Advantage Estimation)
8. On-policy vs Off-policy
9. Experience Replay vs Trajectory
10. Modern RL best practices

---

## 🚀 Sonraki Adımlar

### Diğer Environmentlar:
- **LunarLander** (daha zor, 8D state)
- **MountainCar** (sparse reward problemi)
- **Pendulum** (continuous action space)

### Gelişmiş Algoritmalar:
- **SAC** (Soft Actor-Critic)
- **TD3** (Twin Delayed DDPG)
- **A3C** (Asynchronous Actor-Critic)

### Derin Öğrenme:
- **CNN + RL** (Atari oyunları)
- **RNN + RL** (partial observability)
- **Transformer + RL** (decision transformer)

---

## 🎯 Final Notlar

**DQN ne zaman kullanılır:**
- Discrete action space
- Off-policy learning gerekiyorsa
- Old experiences'ı tekrar kullanmak istiyorsan

**PPO ne zaman kullanılır:**
- Continuous veya discrete actions
- Stable training istiyorsan
- Sample efficiency önemliyse
- State-of-the-art performans istiyorsan

**Başarılarının devamını dilerim!** 🚀🎮

Hazır olduğunda başka projeler ve algoritmalar için devam edebiliriz! 💪
