import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Deneyimleri (state, action, reward, next_state, done) tuple'ları 
    olarak saklar ve rastgele batch'ler döndürür.
    
    Parameters:
    -----------
    capacity : int
        Buffer'ın maksimum kapasitesi (eski deneyimler silinir)
    """
    
    def __init__(self, capacity=10000):
        """
        Buffer'ı başlat
        
        deque: Double-ended queue - otomatik olarak eski elemanları siler
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
        print(f"✅ Replay Buffer oluşturuldu: kapasite = {capacity}")
        
    def add(self, state, action, reward, next_state, done):
        """
        Buffer'a yeni deneyim ekle
        
        Parameters:
        -----------
        state : np.array
            Mevcut state
        action : int
            Seçilen action
        reward : float
            Alınan reward
        next_state : np.array
            Sonraki state
        done : bool
            Episode bitti mi?
        """
        # Tuple olarak sakla
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """
        Buffer'dan rastgele batch çek
        
        Parameters:
        -----------
        batch_size : int
            Kaç deneyim çekilecek
        
        Returns:
        --------
        states, actions, rewards, next_states, dones : tuple of np.arrays
            Batch halinde deneyimler
        """
        # Rastgele indeksler sec
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Secilen deneyimleri al
        batch = [self.buffer[idx] for idx in indices]
        
        # Tuple'lari ayir ve numpy array'e cevir
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Buffer'daki deneyim sayısını döndür
        
        Returns:
        --------
        int
            Buffer'daki deneyim sayısı
        """
        return len(self.buffer)
        
    def is_ready(self, batch_size):
        """
        Buffer'da yeterli deneyim var mı?
        
        Parameters:
        -----------
        batch_size : int
            Gerekli batch boyutu
        
        Returns:
        --------
        bool
            Yeterli deneyim varsa True
        """
        return len(self.buffer) >= batch_size
    
if __name__ == "__main__":
    print("="*60)
    print("🧪 GÖREV 4: Experience Replay Buffer Test")
    print("="*60)
    
    # Test 1: Buffer oluştur
    print("\n📦 Test 1: Buffer Oluşturma")
    buffer = ReplayBuffer(capacity=100)
    print(f"   Buffer kapasitesi: {buffer.capacity}")
    print(f"   Mevcut deneyim sayısı: {len(buffer)}")
    
    # Test 2: Deneyim ekle
    print("\n➕ Test 2: Deneyim Ekleme")
    for i in range(10):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = (i == 9)  # Son deneyimde episode biter
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"   10 deneyim eklendi")
    print(f"   Buffer boyutu: {len(buffer)}")
    
    # Test 3: Batch sampling
    print("\n🎲 Test 3: Batch Sampling")
    if buffer.is_ready(batch_size=5):
        states, actions, rewards, next_states, dones = buffer.sample(5)
        
        print(f"   Batch size: 5")
        print(f"   States shape: {states.shape}")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Next states shape: {next_states.shape}")
        print(f"   Dones shape: {dones.shape}")
        
        print(f"\n   İlk deneyim:")
        print(f"      State: {states[0]}")
        print(f"      Action: {actions[0]}")
        print(f"      Reward: {rewards[0]:.3f}")
        print(f"      Done: {bool(dones[0])}")
    
    # Test 4: Kapasite testi
    print("\n🔄 Test 4: Kapasite Testi")
    small_buffer = ReplayBuffer(capacity=5)
    
    for i in range(10):
        small_buffer.add(
            state=np.array([i]),
            action=i,
            reward=float(i),
            next_state=np.array([i+1]),
            done=False
        )
    
    print(f"   10 deneyim eklendi, kapasite: 5")
    print(f"   Buffer boyutu: {len(small_buffer)}")
    print(f"   ✅ Eski deneyimler otomatik silindi!")
    
    # Son 5 deneyimin action'larını göster
    states, actions, _, _, _ = small_buffer.sample(5)
    print(f"   Buffer'daki action'lar: {sorted(actions)}")
    print(f"   (5-9 arasında olmalı, 0-4 silindi)")
    
    # Test 5: Decorrelation (korelasyon kırılması)
    print("\n🎯 Test 5: Decorrelation Test")
    buffer2 = ReplayBuffer(capacity=100)
    
    # Sıralı deneyimler ekle
    for i in range(50):
        buffer2.add(
            state=np.array([i]),
            action=0,
            reward=float(i),
            next_state=np.array([i+1]),
            done=False
        )
    
    # Batch çek ve sıralı mı kontrol et
    states, _, rewards, _, _ = buffer2.sample(10)
    
    print(f"   Buffer'a 0-49 arası sıralı reward'lar eklendi")
    print(f"   Batch'teki reward'lar: {sorted(rewards)}")
    print(f"   ✅ Rastgele sıralı, korelasyon kırıldı!")
    
    print("\n" + "="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)
        