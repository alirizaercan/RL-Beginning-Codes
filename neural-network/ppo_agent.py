import numpy as np 
from ppo_networks import ActorNetwork, CriticNetwork, compute_advantages

class TrajectoryBuffer:
    """
    PPO için trajectory buffer
    
    Episode trajectory'sini saklar ve batch'ler döndürür
    """
    
    def __init__(self):
        """Buffer'ı başlat"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        """
        Tek adım ekle
        
        Parameters:
        -----------
        state : np.array
            State
        action : int
            Action
        reward : float
            Reward
        done : bool
            Episode bitti mi?
        log_prob : float
            Log probability (π_old)
        value : float
            Value prediction (V(s))
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def get_trajectory(self, last_value, gamma=0.99, lam=0.95):
        """
        Trajectory'yi al ve advantage'ları hesapla
        
        Parameters:
        -----------
        last_value : float
            Son state'in value'su (bootstrap için)
        gamma : float
            Discount factor
        lam : float
            GAE lambda
        
        Returns:
        --------
        states, actions, old_log_probs, advantages, returns : np.arrays
            Training için gerekli data
        """
        # List'leri numpy array'lere cevir
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        log_probs = np.array(self.log_probs)
        values = np.array(self.values).reshape(-1, 1)
        
        # Next values (bootstrap icin)
        next_values = np.vstack([values[1:], [[last_value]]])
        
        # Advantage'lari hesapla (GAE)
        advantages, returns = compute_advantages(
            rewards, values, next_values, dones, gamma, lam
        )
        
        return states, actions, log_probs, advantages, returns
    
    def clear(self):
        """Buffer'ı temizle"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def __len__(self):
        """Buffer'daki adım sayısı"""
        return len(self.states)

class PPOAgent:
    """
    Proximal Policy Optimization Agent
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        hidden_sizes=[128, 64],
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=64
        ):
        """
        PPO Agent'ı başlat
        
        Parameters:
        -----------
        state_size : int
            State boyutu
        action_size : int
            Action sayısı
        hidden_sizes : list
            Hidden layer boyutları
        actor_lr : float
            Actor learning rate
        critic_lr : float
            Critic learning rate
        gamma : float
            Discount factor
        lam : float
            GAE lambda
        clip_epsilon : float
            PPO clip range
        epochs : int
            Her trajectory için kaç epoch eğitim
        mini_batch_size : int
            Mini-batch boyutu
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        
        # Networks
        self.actor = ActorNetwork(
            state_size, action_size, hidden_sizes, actor_lr
        )
        self.critic = CriticNetwork(
            state_size, hidden_sizes, critic_lr
        )
        
        # Trajectory Buffer
        self.buffer = TrajectoryBuffer()
        
        print(f"✅ PPO Agent oluşturuldu!")
        print(f"   Epochs: {epochs}, Mini-batch: {mini_batch_size}")
        print(f"   Clip ε: {clip_epsilon}, γ: {gamma}, λ: {lam}")
        
    def select_action(self, state):
        """
        Action seç ve gerekli bilgileri sakla
        
        Returns:
        --------
        action : int
            Seçilen action
        log_prob : float
            Log probability (π_old için)
        value : float
            Value prediction
        """
        # Actor'dan action
        action, _ = self.actor.get_action(state)
        
        # Log probability (old policy icin)
        log_prob = self.actor.get_log_prob(state, action)
        
        # Value prediction
        value = self.critic.predict(state)[0, 0]
        
        return action, log_prob, value
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """
        Transition'ı buffer'a ekle
        """
        self.buffer.add(state, action, reward, done, log_prob, value)
        
    def train(self):
        """
        Buffer'daki trajectory ile multiple epoch training
        
        Returns:
        --------
        metrics : dict
            Training metrikleri (loss, clip fraction, etc.)
        """
        # Son state icin value (bootstrap)
        # Buffer'da yoksa 0 kullan
        if len(self.buffer) == 0:
            return None
        
        # Son state'in value'sunu tahmin et (bootstrap icin)
        last_state = self.buffer.states[-1]
        last_value = self.critic.predict(last_state.reshape(1, -1))[0, 0]
        
        # Trajectory'yi al
        states, actions, old_log_probs, advantages, returns = \
            self.buffer.get_trajectory(last_value, self.gamma, self.lam)
            
        # Advantage normalization (stability icin)
        advantages = (advantages - np.mean(advantages) / (np.std(advantages) + 1e-8))
        
        trajectory_size = len(states)
        
        # Metrics 
        actor_losses = []
        critic_losses = []
        clip_fractions = []
        
        # multiple epochs 
        for epoch in range(self.epochs):
            # Data'yi shuffle et
            indices = np.random.permutation(trajectory_size)
            
            # Mini-batches 
            for start in range(0, trajectory_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                
                # Mini-batch indices
                batch_indices = indices[start:end]
                
                # Mini-batch data 
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices].reshape(-1, 1)
                
                # Actor update (PPO clip)
                from ppo_networks import compute_ppo_loss
                actor_loss, ratios, clip_frac = compute_ppo_loss(
                    self.actor,
                    batch_states,
                    batch_actions,
                    batch_advantages,
                    batch_old_log_probs,
                    self.clip_epsilon
                )
                
                # Backward pass
                self.actor.backward_ppo(
                    batch_states,
                    batch_actions,
                    batch_advantages,
                    batch_old_log_probs,
                    self.clip_epsilon
                )
                
                # Critic update (MSE)
                batch_values = self.critic.predict(batch_states)
                critic_loss = np.mean((batch_values - batch_returns) ** 2)
                
                self.critic.backward(batch_returns)
                
                # Save metrics
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                clip_fractions.append(clip_frac)
                
        # Buffer'ı temizle
        self.buffer.clear()
        
        # Return metrics
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'clip_fraction': np.mean(clip_fractions)
        }
                
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 9: PPO Training Loop Test")
    print("="*60)
    
    state_size = 4
    action_size = 2
    
    # Test 1: Trajectory Buffer
    print("\n📦 Test 1: Trajectory Buffer")
    buffer = TrajectoryBuffer()
    
    # Fake trajectory ekle
    for i in range(10):
        state = np.random.randn(state_size)
        action = np.random.randint(0, action_size)
        reward = 1.0
        done = (i == 9)
        log_prob = -0.5
        value = 5.0
        
        buffer.add(state, action, reward, done, log_prob, value)
    
    print(f"   Buffer size: {len(buffer)}")
    print(f"   ✅ 10 step eklendi!")
    
    # Trajectory al
    states, actions, log_probs, advantages, returns = buffer.get_trajectory(last_value=0.0)
    
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Advantages shape: {advantages.shape}")
    print(f"   ✅ Trajectory hazırlandı!")
    
    # Test 2: PPO Agent
    print("\n🤖 Test 2: PPO Agent")
    agent = PPOAgent(
        state_size,
        action_size,
        epochs=4,
        mini_batch_size=32
    )
    
    # Action seç
    state = np.random.randn(state_size)
    action, log_prob, value = agent.select_action(state)
    
    print(f"   Action: {action}")
    print(f"   Log prob: {log_prob:.4f}")
    print(f"   Value: {value:.4f}")
    print(f"   ✅ Action selection çalışıyor!")
    
    # Test 3: Training
    print("\n🎓 Test 3: Training Loop")
    
    # Trajectory topla
    for i in range(64):  # 64 step
        state = np.random.randn(state_size)
        action, log_prob, value = agent.select_action(state)
        reward = 1.0
        done = False
        
        agent.store_transition(state, action, reward, done, log_prob, value)
    
    print(f"   Buffer: {len(agent.buffer)} steps")
    
    # Train
    metrics = agent.train()
    
    print(f"   Actor loss: {metrics['actor_loss']:.4f}")
    print(f"   Critic loss: {metrics['critic_loss']:.4f}")
    print(f"   Clip fraction: {metrics['clip_fraction']:.2%}")
    print(f"   ✅ Training tamamlandı!")
    
    # Buffer temizlendi mi?
    print(f"   Buffer size after train: {len(agent.buffer)}")
    
    # Test 4: Multiple epochs
    print("\n🔁 Test 4: Multiple Epochs Effect")
    
    # Yeni agent
    agent_multi = PPOAgent(state_size, action_size, epochs=5, mini_batch_size=16)
    
    # Fake trajectory
    for i in range(48):
        state = np.random.randn(state_size)
        action, log_prob, value = agent_multi.select_action(state)
        agent_multi.store_transition(state, action, 1.0, False, log_prob, value)
    
    # Metrics before
    state_test = np.random.randn(1, state_size)
    value_before = agent_multi.critic.predict(state_test)[0, 0]
    
    # Train
    metrics = agent_multi.train()
    
    # Metrics after
    value_after = agent_multi.critic.predict(state_test)[0, 0]
    
    print(f"   Epochs: 5")
    print(f"   Value (önce): {value_before:.4f}")
    print(f"   Value (sonra): {value_after:.4f}")
    print(f"   ✅ Multiple epochs policy'yi güncelledi!")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)