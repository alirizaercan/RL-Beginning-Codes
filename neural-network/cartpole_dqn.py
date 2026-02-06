import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt 
from collections import deque
from q_learning_nn import QNetwork
from experience_replay import ReplayBuffer

class FrameStack:
    """
    Frame Stacking Buffer
    
    Son N frame'i birleştirerek state'i zenginleştir
    Örn: num_frames=4, state_size=4 → output=16 boyut
    """
    
    def __init__(self, num_frames=1, state_size=4):
        """
        Parameters:
        -----------
        num_frames : int
            Kaç frame stack et (1=no stacking, 2=2 frame, 4=4 frame)
        state_size : int
            Tek frame'in boyutu (CartPole=4)
        """
        self.num_frames = num_frames
        self.state_size = state_size
        self.stacked_size = num_frames * state_size  # Output boyutu
        self.buffer = deque(maxlen=num_frames)
        
    def reset(self, state):
        """Episode başlat: buffer'ı ilk state ile doldur"""
        self.buffer.clear()
        for _ in range(self.num_frames):
            self.buffer.append(state.copy())
    
    def add(self, state):
        """Yeni frame ekle"""
        self.buffer.append(state.copy())
    
    def get_stacked(self):
        """Son N frame'i concatenate et"""
        return np.concatenate(list(self.buffer))
    
    def __repr__(self):
        return f"FrameStack(num_frames={self.num_frames}, stacked_size={self.stacked_size})"

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
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Network (learning rate dinamik olarak ayarla - state_size'a göre)
        # Daha büyük input → daha düşük learning rate
        dynamic_lr = learning_rate / (1 + state_size / 10)
        
        self.q_network = QNetwork(
            state_size,
            action_size,
            hidden_size,
            learning_rate=dynamic_lr
        )
        
        # Experience Replay Buffer
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
        # Yeterli deneyim var mi? 
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Batch sample
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
            
        # Current Q-values
        current_q_values = self.q_network.predict(states)
        
        # Next Q-values (target icin)
        next_q_values = self.q_network.predict(next_states)
        
        # Target Q-values hesapla (Bellman Equation)
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                # Episode bittiyse sadece reward
                target_q_values[i, actions[i]] = rewards[i]
            else: 
                # Bellman equation: Q(s,a) = r + γ * max_a' Q(s',a')
                target_q_values[i, actions[i]] = \
                    rewards[i] + self.gamma * np.max(next_q_values[i])
                    
        # Backward pass (network'u guncelle)
        self.q_network.backward(target_q_values)
        
        # Loss hesapla 
        loss = np.mean((current_q_values - target_q_values) ** 2)
        
        return loss
    
    def decay_epsilon(self):
        """
        Epsilon'u azalt (exploration → exploitation)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
def train_dqn(episodes=500, render=False, num_frames=1):
    """
    DQN ile CartPole eğit (Frame Stacking ile, Optimized)
    
    Parameters:
    -----------
    episodes : int
        Kaç episode eğitim yapılacak
    render : bool
        Ortamı görselleştir mi?
    num_frames : int
        Frame stack sayısı (1=no stack, 2=2-frame, 4=4-frame)
    """
    # Ortam olustur
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # Frame Stack buffer
    frame_stack = FrameStack(num_frames=num_frames, state_size=4)
    
    # Dinamik hyperparameter ayarı
    # Input boyutu arttıkça hidden layer'ı da arttır
    input_size = 4 * num_frames
    hidden_size = 64 * max(1, input_size // 4)  # 4→64, 8→128, 16→256
    
    # Agent olustur (state_size = 4 * num_frames)
    agent = DQNAgent(
        state_size=input_size,
        action_size=2,
        hidden_size=hidden_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64
    )
    
    # Metrikleri sakla 
    episode_rewards = []
    episode_losses = []
    
    print("\n" + "="*60)
    print(f"🚀 Eğitim Başlıyor! (Frame Stack: {frame_stack})")
    print("="*60)
    
    for episode in range(episodes):
        # Episode baslat
        raw_state, _ = env.reset()
        frame_stack.reset(raw_state)  # Buffer'ı ilk state ile doldur
        state = frame_stack.get_stacked()  # Stacked state
        
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Action sec (stacked state kullan)
            action = agent.select_action(state)
            
            # Environment'e action sec
            raw_next_state, reward, terminated, truncated, _ = env.step(action)
            frame_stack.add(raw_next_state)  # Yeni frame ekle
            next_state = frame_stack.get_stacked()  # Stacked state
            done = terminated or truncated
            
            # Deneyimi sakla (stacked states)
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train et (eger yeterli deneyim varsa)
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)
                
            episode_reward += reward 
            state = next_state
        
        # Epsilon decay
        agent.decay_epsilon()
        
        # Metrikleri kaydet
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # Ilerleme goster
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1:3d} | "
                f"Reward: {episode_reward:3.0f} | "
                f"Avg(10): {avg_reward:.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {agent.epsilon:.3f}")
            
        # Hedef basari kontrolu
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            if avg_100 >= 195.0:
                print(f"\n🎉 Çözüldü! Episode {episode+1}'de")
                print(f"   Son 100 episode ortalaması: {avg_100:.1f}")
                break
        
    env.close()
    
    return episode_rewards, episode_losses, agent

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
    plt.savefig('cartpole_dqn_results.png')
    print(f"\n📊 Grafik kaydedildi: cartpole_dqn_results.png")
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
    
if __name__ == "__main__":
    # Eğitim
    print("🎮 CartPole DQN - Frame Stacking ile Eğitim")
    
    # Hangi frame stack'i test edelim?
    num_frames = 4  # 1, 2, 4 deneyelim
    
    print(f"\n🎬 Frame Stack Sayısı: {num_frames}")
    print(f"   Input boyutu: {4 * num_frames}")
    
    rewards, losses, agent = train_dqn(episodes=500, render=False, num_frames=num_frames)
    
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