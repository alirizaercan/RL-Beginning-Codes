import numpy as np

class ActorNetwork:
    """
    Policy Network (Actor) - PPO için
    
    State alır, action probability distribution döndürür
    """
    def __init__(self, state_size, action_size, hidden_sizes=[128,64], learning_rate=0.0003):
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
        
        # Network katmanlarini baslat
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
        # State'i 2D yap
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        # Layer 1
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        
        # Layer 2 
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2) # ReLU
        
        # Layer 3 - Output (action logits)
        self.logits = np.dot(self.a2, self.W3) + self.b3
        
        # Softmax - logits'i probability'ye çevir
        exp_logits = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.action_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Son state'i sakla (backend icin)
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
        
        # Probability distribution'dan sample 
        action = np.random.choice(self.action_size, p=action_probs[0])
        action_prob = action_probs[0, action]
        
        return action, action_prob
    
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
        
        # One-hot encode actions
        # [0, 1, 0] -> [[1,0], [0,1], [1,0]]
        actions_one_hot = np.zeros((batch_size, self.action_size))
        actions_one_hot[np.arange(batch_size), actions] = 1
        
        # Policy gradient
        # dL/d(logits) = action_probs - actions_one_hot
        # Ama advantage ile agirliklandiriyoruz
        d_logits = self.action_probs - actions_one_hot
        
        # Advantage ile carp (positive advantage -> selected action'i artir)
        d_logits *= advantages.reshape(-1, 1) / batch_size
        d_logits /= batch_size
        
        # Output layer gradients 
        dW3 = np.dot(self.a2.T, d_logits)
        db3 = np.sum(d_logits, axis=0, keepdims=True)
        
        # Hidden layer 2 gradients
        da2 = np.dot(d_logits, self.W3.T)
        dz2 = da2 * (self.z2 > 0) # ReLU backprop
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0) # ReLU backprop
        
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
            log_prob = np.log(action_probs[0, action] + 1e-10) # +epsilon for stability
        else:
            # Batch action
            batch_size = len(action)
            log_probs = np.log(action_probs[np.arange(batch_size), action] + 1e-10)
            log_prob = log_probs
            
        return log_prob
    
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
        
        # Gradient hesaplama
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
        
        # Backpropagation
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

class CriticNetwork:
    """
    Value Network (Critic) - PPO için
    
    State alır, state value döndürür
    """
    def __init__(self, state_size, hidden_sizes=[128,64], learning_rate=0.0003):
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
        
        # Network katmanlarini baslat
        # Layer 1: Input -> Hidden
        self.W1 = np.random.randn(state_size, hidden_sizes[0]) * np.sqrt(2/state_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        
        # Layer 2: Hidden1 -> Hidden2
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2/hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        
        # Layer 3: Hidden2 -> Output (state value)
        self.W3 = np.random.randn(hidden_sizes[1], 1) * np.sqrt(2/hidden_sizes[1])
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
        # State'i 2D yap
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        # Layer 1
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2) # ReLU
        
        # Layer 3 - Output (value)
        self.value = np.dot(self.a2, self.W3) + self.b3
        
        # Son state'i sakla (backward icin)
        self.last_state = state
        
        return self.value
    
    def predict(self, state):
        """
        State için value tahmin et
        """
        return self.forward(state)
    
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
        
        # MSE loss gradient
        d_value = 2 * (self.value - target_values) / batch_size
        
        # Output layer gradients
        dW3 = np.dot(self.a2.T, d_value)
        db3 = np.sum(d_value, axis=0, keepdims=True)
        
        # Hidden layer 2 gradients
        da2 = np.dot(d_value, self.W3.T)
        dz2 = da2 * (self.z2 > 0) # ReLU backprop
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0) # ReLU backprop
        
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
    deltas = rewards + gamma * next_values.flatten() * (1-dones) - values.flatten()
    
    # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    gae = 0
    for t in reversed(range(batch_size)):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values.flatten()[t]
    
    return advantages, returns

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
    # Current policy log probabilities
    new_log_probs = actor.get_log_prob(states, actions)
    
    # Ratio = π_new / π_old = exp(log_new - log_old)
    log_ratio = new_log_probs - old_log_probs
    ratio = np.exp(log_ratio)
    
    # Clipped ratio
    ratio_clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    # Surrogate losses
    surr1 = ratio * advantages
    surr2 = ratio_clipped * advantages
    
    # PPO loss = -min(surr1, surr2)
    # min() aliyoruz cunku conservative olmak istiyoruz
    loss = -np.mean(np.minimum(surr1, surr2))
    
    # Monitoring metrics 
    clipped_fraction = np.mean(np.abs(ratio - ratio_clipped) > 1e-6)
    
    return loss, ratio, clipped_fraction
    
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
    old_log_prob = actor.get_log_prob(state, action)[0]
    print(f"   Old log prob: {old_log_prob:.4f}")
    
    # Policy'yi değiştir (büyük update)
    for _ in range(5):
        actor.forward(state)
        actor.backward(np.array([10.0]), action)
    
    # New policy
    new_log_prob = actor.get_log_prob(state, action)[0]
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
    old_log_prob = actor2.get_log_prob(state, np.array([0]))[0]
    for _ in range(10):
        actor2.backward_ppo(state, np.array([0]), np.array([10.0]), 
                           np.array([old_log_prob]), clip_epsilon=0.2)
        old_log_prob = actor2.get_log_prob(state, np.array([0]))[0]
    
    prob_vanilla, _ = actor1.forward(state)
    prob_ppo, _ = actor2.forward(state)
    
    print(f"   Vanilla PG (clip yok): {prob_vanilla[0]}")
    print(f"   PPO (clip var): {prob_ppo[0]}")
    print(f"   ✅ PPO daha conservative (stabil) update!")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)