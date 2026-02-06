import numpy as np
import gymnasium as gym

class SimpleAgent:
    def __init__(self, state_size=8, action_size=2):
        # input_size=8 çünkü [t ve t-1] birleşik
        self.weights = np.random.randn(state_size, action_size) * 0.1

    def get_action(self, stacked_state):
        q_values = np.dot(stacked_state, self.weights)
        return np.argmax(q_values), q_values

# 1. ortamı kur
env = gym.make("CartPole-v1", render_mode="human")
agent = SimpleAgent(state_size=8, action_size=2)

# 2. başlangıç verileri
state, _ = env.reset()
prev_state = state
total_reward = 0

print(f"{'STEP':<6} | {'ACTION':<8} | {'REWARD':<7} | {'TOTAL':<7} | {'STABILITY (Angle Diff)':<20}")
print("-" * 65)

for step in range(200):
    # --- XYZ MANTIGI: FRAME STACKING ---
    stacked_state = np.concatenate([state, prev_state]) 
    
    # 3. aksiyon seç ve Q değerlerini gör
    action, q_vals = agent.get_action(stacked_state)
    
    # Analiz: Direğin açısal değişimi (türev/ivme belirtisi)
    # state[2] o anki açıdır, prev_state[2] bir önceki.
    angle_diff = state[2] - prev_state[2]
    
    # 4. dünyada hareket et
    prev_state = state.copy() # hafızaya al
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    
    # --- PRINT BÖLÜMÜ ---
    action_str = "SAĞ (1)" if action == 1 else "SOL (0)"
    print(f"{step+1:<6} | {action_str:<8} | {reward:<7.1f} | {total_reward:<7.1f} | {angle_diff:<20.4f}")
    
    if terminated or truncated:
        print(f"\n💥 EPİSODE BİTTİ! Toplam Skor: {total_reward}")
        print("="*65 + "\n")
        state, _ = env.reset()
        prev_state = state
        total_reward = 0

env.close()