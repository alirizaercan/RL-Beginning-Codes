from collections import deque
import numpy as np 
import gymnasium as gym 

class FrameStackingWrapper:
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

        # Ilk frame'i on-isle
        preprocessed = self._preprocess_frame(observation)

        # Stack'i bos frame'lerle doldur
        for _ in range(self.num_frames):
            self.frames.append(preprocessed)

        # Stacked frame'leri dondur
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

        # YEni frame'i om isle ve stack'e ekle
        preprocessed = self._preprocess_frame(observation)
        self.frames.append(preprocessed)

        # Stacked frame'leri dondur
        stacked = self._get_stacked_frames()

        return stacked, reward, terminated, truncated, info
    
    def _preprocess_frame(self, frame):
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

        # Cubuk pozisyonunu state'dan al
        # observation: [x, y, θ, ω]
        x, y, theta, omega = observation

        # Normalizasyon (CartPole bounds bilinir)
        x_norm = (x + 2.4) / 4.8  # x ∈ [-2.4, 2.4]
        theta_norm = (theta + 0.2) / 0.4  # θ ∈ [-0.2, 0.2]

        # Cubuk ciz (basit line)
        pole_lenth = 20
        pole_end_x = int(center_x + pole_length * np.sin(theta))
        pole_end_y = int(center_y - pole_length * np.cos(theta))
    
        # Bresenham line algoritmasi ile ciz
        canvas = self._draw_line(canvas, center_x, center_y, 
                                 pole_end_x, pole_end_y, value=1.0)
        
        # Normalize et [0, 1]
        frame = canvas / 255.0

        return frame
    
    