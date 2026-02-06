import numpy as np

class QNetwork:
    """
    Q-Learning için Neural Network
    
    Parametreler:
    - state_size: State vektörünün boyutu (örn: CartPole'da 4)
    - action_size: Kaç farklı action var (örn: CartPole'da 2)
    - hidden_size: Hidden layer'da kaç nöron (default: 64)
    - learning_rate: Öğrenme hızı (default: 0.001)
    """
    
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # Ağırlıkları He initialization ile başlat
        
        # Layer 1 : Input -> Hidden
        self.W1 = np.random.randn(state_size, hidden_size) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Layer 2 : Hidden -> Output
        self.W2 = np.random.randn(hidden_size, action_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, action_size))
        
        print(f"✅ Ağırlıklar başlatıldı!")
        print(f"   W1 shape: {self.W1.shape} - Input → Hidden")
        print(f"   W2 shape: {self.W2.shape} - Hidden → Output")
        
    def forward(self, state):
        """
        Forward pass - State'den Q-value'lara git
        
        Parameters:
        -----------
        state : np.array
            Shape: (state_size,) veya (batch_size, state_size)
            Örnek: [0.1, 0.5, -0.2, 0.3] - CartPole state'i
        
        Returns:
        --------
        q_values : np.array
            Her action için Q-value tahmini
            Shape: (1, action_size) veya (batch_size, action_size)
        """
        # State'in 2D yapiyoruz
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        # Layer 1: Input -> Hidden
        # z1 = state @ W1 + b1
        self.z1 = np.dot(state, self.W1) + self.b1
        
        # ReLu aktivasyonu
        # a1 = max(0, z1)
        self.a1 = np.maximum(0, self.z1)
        
        # Layer 2 : Hidden -> Output
        # q_values = a1 @ W2 + b2
        self.q_values = np.dot(self.a1, self.W2) + self.b2
        
        # Son state'i sakla (backward pass icin lazim)
        self.last_state = state
        
        return self.q_values
    
    def predict(self, state):
        """
        State için Q-values tahmin et
        
        Parameters:
        -----------
        state : np.array
            State vektörü
        
        Returns:
        --------
        q_values : np.array
            Her action için Q-value
        """
        return self.forward(state)
    
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
        
        # Loss gradient - MSE'nin turevi
        # dL/dQ = 2 * (predicted - target) / batch_size
        dq = 2 * (self.q_values - target_q_values) / batch_size
        
        # Output layer gradients
        # dL/dW2 = a1.T @ dq
        # dL/db2 = sum(dq)
        dW2 = np.dot(self.a1.T, dq)
        db2 = np.sum(dq, axis=0, keepdims=True)
        
        # Hidden layer gradients
        # Once: dL/da1 = dq @ W2.T
        da1 = np.dot(dq, self.W2.T)
        
        # ReLu backprop: sadece pozitif olan yerlerde gradient gecer
        # dL/dz1 = da1 * (z1 > 0)
        dz1 = da1 * (self.z1 > 0)
        
        # Input layer gradients
        # dL/dW1 = state.T @ dz1
        # dL/db1 = sum(dz1)
        dW1 = np.dot(self.last_state.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Gradient clipping - Büyük gradient'leri sınırla (stability için)
        max_grad = 1.0
        dW1 = np.clip(dW1, -max_grad, max_grad)
        dW2 = np.clip(dW2, -max_grad, max_grad)
        db1 = np.clip(db1, -max_grad, max_grad)
        db2 = np.clip(db2, -max_grad, max_grad)
        
        # Gradient descent ile ağırlıkları güncelle
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
    def get_action(self, state, epsilon=0.0):
        """
        Epsilon-greedy action seçimi
        
        Parameters:
        -----------
        state : np.array
            Mevcut state
        epsilon : float
            Exploration oranı (0-1 arası)
            0.0: Tamamen exploit (greedy)
            1.0: Tamamen explore (random)
        
        Returns:
        --------
        action : int
            Seçilen action index
        """
        # Epsilon olasilikla random action
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_size)
        
        # (1-epsilon) olasilikla en iyi action
        q_values = self.predict(state)
        return np.argmax(q_values)
    
if __name__ == "__main__":
    STACK_SIZE = 2
    ORIGINAL_STATE = 4
    qnet = QNetwork(state_size=ORIGINAL_STATE * STACK_SIZE, action_size=2, hidden_size=32, learning_rate=0.1)
    print(f"✅ Q-Network oluşturuldu: {qnet.state_size} → {qnet.hidden_size} → {qnet.action_size}")
    
    print(f"\n🔍 W1 shape: {qnet.W1.shape}")
    print(f"🔍 W2 shape: {qnet.W2.shape}")
    print(f"🔍 b1 shape: {qnet.b1.shape}")
    print(f"🔍 b2 shape: {qnet.b2.shape}")
    
    print("="*60)
    print("🧪 GÖREV 2: Forward Pass Test")
    print("="*60)
    
    # Test 1: Tek state (frame stacking ile)
    print("\n📍 Test 1: Tek State (Frame Stacking)")
    single_frame = np.array([0.1, 0.5, -0.2, 0.3])
    # Frame stack: 2 frame birleştir (simülasyon için aynı frame'i kopyala)
    state = np.concatenate([single_frame, single_frame])  # 4+4=8 boyut
    print(f"   Tek Frame: {single_frame} (shape: {single_frame.shape})")
    print(f"   Stacked State: {state}")
    print(f"   Stacked State Shape: {state.shape}")
    
    q_values = qnet.predict(state)
    print(f"   Output Q-Values: {q_values[0]}")
    print(f"   Q-Values Shape: {q_values.shape}")
    print(f"   En iyi Action: {np.argmax(q_values)}")
    
    # Test 2: Batch processing (birden fazla state)
    print("\n📦 Test 2: Batch Processing")
    # Her frame 4 boyut, stack=2 olduğu için her state 8 boyut
    batch_states = np.array([
        [0.1, 0.5, -0.2, 0.3, 0.1, 0.5, -0.2, 0.3],     # State 1 (frame_t + frame_t-1)
        [-0.3, 0.2, 0.1, -0.5, -0.3, 0.2, 0.1, -0.5],   # State 2
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        # State 3
    ])
    print(f"   Batch Shape: {batch_states.shape}")
    
    batch_q_values = qnet.predict(batch_states)
    print(f"   Batch Q-Values Shape: {batch_q_values.shape}")
    print(f"   Q-Values:")
    for i, q in enumerate(batch_q_values):
        print(f"      State {i}: {q} → Action: {np.argmax(q)}")
    
    # Test 3: ReLU'nun çalıştığını göster
    print("\n🔬 Test 3: ReLU Aktivasyonu")
    print(f"   Hidden Layer (z1) - İlk 5 değer:")
    print(f"      Önce: {qnet.z1[0, :5]}")
    print(f"      Sonra (ReLU): {qnet.a1[0, :5]}")
    print(f"   ✅ Negatif değerler 0 oldu!")
    
    # Test 4: Aynı state her zaman aynı Q-value verir mi?
    print("\n🔁 Test 4: Determinism (Belirlilik)")
    q1 = qnet.predict(state)
    q2 = qnet.predict(state)
    print(f"   İlk tahmin: {q1[0]}")
    print(f"   İkinci tahmin: {q2[0]}")
    print(f"   Aynı mı? {np.allclose(q1, q2)}")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
    
    
    print("="*60)
    print("🧪 GÖREV 3: Backward Pass & Öğrenme Testi")
    print("="*60)
    
    STACK_SIZE = 2
    ORIGINAL_STATE = 4
    qnet = QNetwork(state_size=ORIGINAL_STATE * STACK_SIZE, action_size=2, hidden_size=32, learning_rate=0.1)
    
    # Test state
    state = np.array([0.1, 0.5, -0.2, 0.3])
    
    # Test 1: Öğrenmeden önce
    print("\n📊 Test 1: Öğrenmeden Önce")
    # Frame stacking ile state oluştur
    single_frame = np.array([0.1, 0.5, -0.2, 0.3])
    state = np.concatenate([single_frame, single_frame])  # 8 boyut
    q_before = qnet.predict(state)
    print(f"   State (stacked): {state}")
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
    qnet2 = QNetwork(state_size=ORIGINAL_STATE * STACK_SIZE, action_size=2, hidden_size=32, learning_rate=0.01)
    
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
    
    
    # Epsilon greedy test
    print("\n🎲 Epsilon-Greedy Test:")
    single_frame = np.array([0.1, 0.5, -0.2, 0.3])
    state = np.concatenate([single_frame, single_frame])  # 8 boyut stacked state

    for eps in [0.0, 0.5, 1.0]:
        actions = [qnet.get_action(state, epsilon=eps) for _ in range(100)]
        action_counts = np.bincount(actions, minlength=2)
        print(f"   ε={eps:.1f}: Action 0: {action_counts[0]}%, Action 1: {action_counts[1]}%")
        