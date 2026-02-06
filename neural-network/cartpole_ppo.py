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
    # Environmet 
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # PPO Agent
    agent = PPOAgent(
        state_size=4,
        action_size=2,
        hidden_sizes=[128,64],
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
        # Action sec
        action, log_prob, value = agent.select_action(state)
        
        # Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(state, action, reward, done, log_prob, value)
        
        episode_reward += reward
        state = next_state
        steps += 1
        
        # Episode bitti mi?
        if done: 
            episode_count += 1
            episode_rewards.append(episode_reward)
            
            # Ilerleme goster
            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode_count:3d} | "
                      f"Reward: {episode_reward:3.0f} | "
                      f"Avg(10): {avg_reward:.1f} | "
                      f"Steps: {steps}")
                
            # Reset
            state, _ = env.reset()
            episode_reward = 0
            
            # Basari kontrolu
            if len(episode_rewards) >= 100:
                avg_100 = np.mean(episode_rewards[-100:])
                if avg_100 >= 195:
                    print(f"\n🎉 Çözüldü! Episode {episode_count}'de")
                    print(f"   Son 100 episode ortalaması: {avg_100:.1f}")
                    break
                
        # Train et (trajectory doldu mu?)
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
                
            # Buffer temizlendi, yeni trajectory'e basla
            steps = 0
        
        # Max episode kontrolu
        if episode_count >= episodes:
            break
        
    env.close()
    
    return episode_rewards, all_actor_losses, all_critic_losses, all_clip_fractions, agent

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