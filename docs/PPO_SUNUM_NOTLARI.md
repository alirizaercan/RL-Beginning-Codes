# PPO LunarLander Eğitimi - Sunum Notları

## 1. PROJE HAKKINDA

**Amaç:** OpenAI Gymnasium'daki LunarLander-v3 ortamında PPO (Proximal Policy Optimization) ile otonom iniş öğretmek.

**Ortam:** LunarLander-v3
- **Observation space:** 8 boyutlu (x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact)
- **Action space:** 4 ayrık aksiyon (do nothing, fire left, fire main, fire right)
- **Reward:** İniş başarısı +100, crash -100, yakıt kullanımı -, yere değmek +10/bacak

---

## 2. PPO ALGORİTMASI NEDIR?

PPO = **Proximal Policy Optimization** (2017, OpenAI)

### Temel Fikir:
- **On-policy** algoritma: Her update'te yeni data topla
- **Actor-Critic:** Policy (actor) + Value function (critic)
- **Clipping:** Policy'nin çok hızlı değişmesini engelle (stability)

### Akış:
```
1. Rollout: 1024 step environment interaction
2. GAE: Advantage hesapla (hangi aksiyonlar iyiydi?)
3. Policy Update: PPO loss ile policy güncelle (clipped)
4. Value Update: Value function MSE loss ile güncelle
5. Tekrar 1'e dön
```

### Neden PPO?
✅ **Stable:** TRPO kadar kararlı ama daha basit  
✅ **Sample efficient:** A3C'den daha iyi  
✅ **Versatile:** Hem discrete hem continuous actions  

---

## 3. KOD MİMARİSİ

### A. Network (ActorCritic)
```python
Input (8) → [128] Tanh → [128] Tanh → {Policy Head (4), Value Head (1)}
```

**Shared layers:** Policy ve value aynı feature extractor kullanır (öğrenme hızı artar)

**Initialization:**
- Shared: Orthogonal init, gain=√2 (RL için best practice)
- Policy: Orthogonal init, gain=0.01 (küçük başlangıç, exploration)
- Value: Orthogonal init, gain=1.0 (normal başlangıç)

### B. GAE (Generalized Advantage Estimation)
```python
Advantage = Σ (γλ)^t * δ_t
δ_t = r_t + γV(s_{t+1}) - V(s_t)  # TD error
```

**Amaç:** Hangi aksiyonlar expected return'den daha iyi?  
**λ = 0.95:** Bias-variance trade-off (yüksek λ → long-term focus)

### C. PPO Loss
```python
L_CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
r_t = π_new(a|s) / π_old(a|s)  # probability ratio
```

**Clipping (ε=0.2):** Ratio çok büyükse/küçükse kes → stability

### D. Value Loss
```python
L_VF = (V_pred - V_target)^2
```

**Value clipping:** Aynı PPO gibi, value çok değişmesin

---

## 4. KARŞILAŞTIĞIMIZ PROBLEMLER & ÇÖZÜMLER

### Problem 1: **Value Function Patlaması**
**Belirti:**
- Value loss = 1000-3000 (normal ~1-10 olmalı)
- Episode rewards hiç iyileşmiyor (-150 civarı takılı)
- Policy loss ~0 (hiç update yok)

**Kök Sebep:**
- Value network reward scale'ini öğrenemiyor
- Large negative rewards (-500) → value tahminleri çarpık
- Advantage signals bozuk → policy update yok

**Çözüm:**
```python
VALUE_LR = 1e-3  # Policy'den 4x daha yüksek
MAX_VALUE_LOSS = 10.0  # Extreme losses clip et
VALUE_COEF = 1.0  # Value loss priority
```

### Problem 2: **Entropy Collapse**
**Belirti:**
- Entropy: 1.38 → 1.27 (hızla düşüyor)
- Policy tek bir aksiyonu seçip takılı kalıyor

**Kök Sebep:**
- `ENTROPY_COEF = 0.01` çok düşük
- Network çok erken deterministic oluyor

**Çözüm:**
```python
ENTROPY_COEF = 0.01  # Başlangıçta 0.05 denedik ama fazlaydı
# Final: 0.01 yeterli (network yeterince explore ediyor)
```

### Problem 3: **Policy Hiç Güncellenmiyor**
**Belirti:**
- `approx_kl ≈ 0.0001` (çok düşük)
- `clipfrac = 0.0` (clipping yok)
- Policy loss ~0

**Kök Sebep:**
- Learning rate çok düşük (1e-4)
- Advantage signal zayıf (value function kötü)

**Çözüm:**
```python
POLICY_LR = 2.5e-4  # 1e-4'ten arttırdık
# Value fix edince advantage signals düzeldi
```

### Problem 4: **Network Capacity Yetersiz**
**Belirti:**
- 64 neuron başlangıçta yetersiz kaldı

**Çözüm:**
```python
# 64 → 128 neurons
nn.Linear(obs_dim, 128)
nn.Linear(128, 128)
```

### Problem 5: **Return Normalization Zararlı**
**Belirti:**
- Returns normalize edince value targets bozuluyor

**Çözüm:**
```python
# Returns NORMALIZE ETME (sadece advantages normalize et)
advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
returns = advantages + values  # RAW returns kullan
```

---

## 5. METRIK ANALİZİ (Log/Grafik)

### A. Episode Reward
**Ne ölçer:** Her bölümün toplam ödülü  
**Hedef:** -150'den +200'e yükselmeli  

**İlk durumlar:**
- Update 1-10: -200 ile -300 arası → rastgele crash  
- Update 20-30: -150 civarı → biraz öğrenme başladı  
- **Başarı kriteri:** +200 üzeri (iniş başarılı)

### B. Policy Loss
**Ne ölçer:** Policy güncellemesinin büyüklüğü  
**Normal değer:** 0.001-0.01 arası  

**Sorunlu:** ~0 ise policy güncellenmiyor  
**İyi:** Negatif değerler OK (maximize reward)

### C. Value Loss
**Ne ölçer:** Value function tahmin hatası  
**Normal değer:** 1-10 arası  

**İlk sorunlar:** 1000-3000 (patlamış)  
**Düzeltmeden sonra:** ~5-10 (iyi)

### D. Entropy
**Ne ölçer:** Policy'nin keşif gücü  
**Normal değer:** 1.3-1.4 (başlangıç), 0.8-1.0 (sonraki)  

**Çok düşük (0.5):** Policy collapse etti  
**İyi:** Yavaşça azalmalı (1.38 → 1.0)

### E. Approx KL
**Ne ölçer:** Policy değişim büyüklüğü  
**Hedef:** 0.001-0.01 arası (sağlıklı update)  

**Çok düşük:** Policy güncellenmiyor  
**Çok yüksek (>0.05):** Instability riski

### F. Clip Fraction
**Ne ölçer:** Ne kadar clipping yapıldı?  
**Normal:** 0.1-0.3 arası  

**0.0:** Clipping yok → update çok küçük  
**0.5+:** Çok fazla clip → learning yavaş

---

## 6. HİPERPARAMETRE SEÇİMLERİ

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **GAMMA** | 0.99 | Discount factor (long-term focus) |
| **LAMBDA** | 0.95 | GAE lambda (bias-variance) |
| **CLIP_EPS** | 0.2 | PPO clipping threshold |
| **POLICY_LR** | 2.5e-4 | Policy learning rate |
| **VALUE_LR** | 1e-3 | Value network LR (daha yüksek!) |
| **ENTROPY_COEF** | 0.01 | Exploration bonus |
| **VALUE_COEF** | 1.0 | Value loss weight |
| **BATCH_SIZE** | 256 | Minibatch size |
| **TIMESTEPS** | 1024 | Rollout length |
| **EPOCHS** | 4 | Update epochs per rollout |
| **MAX_GRAD_NORM** | 0.5 | Gradient clipping |

---

## 7. DEBUGGING SÜRECI

### İlk Deneme: Naive PPO
- ❌ Value loss 2000+
- ❌ Rewards -200 takılı
- ❌ 1.5 saat çalıştı, hiç öğrenmedi

### İkinci Deneme: Entropy + Network Size
- ✅ Entropy coef 0.05'e çıkarıldı
- ✅ Network 64 → 128
- ⚠️ Value loss hala yüksek (~700)
- ⚠️ Yavaş öğrenme

### Üçüncü Deneme: Value Learning Fix
- ✅ Separate value LR (1e-3)
- ✅ Value loss clipping (max=10)
- ✅ VALUE_COEF = 1.0
- ✅ Returns normalize kaldırıldı
- **BAŞARILI:** Value loss 5-10'a düştü

### Son Durum:
- ✅ Value function düzgün öğreniyor
- ✅ Policy updates sağlıklı
- 🔄 ~30 dakikada anlamlı öğrenme görülüyor

---

## 8. SONUÇ & ÖĞRENMELER

### Teknik Öğrenmeler:
1. **Value function critical:** PPO'da value iyi öğrenmezse tüm sistem çöker
2. **Separate LRs:** Policy ve value farklı hızlarda öğrenmeli
3. **Loss clipping:** Extreme losses numerical instability yaratır
4. **Return normalization zararlı:** Value targets'ı bozar
5. **Entropy management:** Çok erken collapse önlenmeli ama fazla da olmamalı

### Pratik Öğrenmeler:
1. **Logging essential:** Her metriği kaydet, sonra analiz et
2. **Start simple:** İlk küçük network/LR ile test et
3. **Value loss priority:** İlk value'yu fix et, sonra policy
4. **Patience:** RL eğitimi zaman alır (30+ dakika)

---

## 9. SONRAKI ADIMLAR

### Kısa Vadeli:
- ✅ JSBSIM entegrasyonu (bugün)
- ✅ FlightGear visual (hafta sonu)

### Uzun Vadeli:
- 🎯 Continuous control (uçak için)
- 🎯 Multi-agent PPO
- 🎯 Sim-to-real transfer

---

## 10. PROFESÖRLERE SORULACAK SORULAR

1. **Value learning rate:** Separate LR kullanımı best practice mi?
2. **Return normalization:** Hangi durumlarda yararlı/zararlı?
3. **GAE lambda:** 0.95 vs 0.97 vs 0.99 trade-off'ları?
4. **JSBSIM için:** Continuous action space'de PPO'da değişiklik gerekir mi?
5. **Sim-to-real:** Simülasyonda öğrenilenler gerçek drone'da çalışır mı?

---

## HIZLI REFERANS: KOD BLOKLARI

### Actor-Critic Forward:
```python
logits, value = model(obs)  # Policy logits ve value prediction
dist = Categorical(logits=logits)  # Action distribution
action = dist.sample()  # Sample action
log_prob = dist.log_prob(action)  # For PPO loss
```

### PPO Loss:
```python
ratio = torch.exp(new_logp - old_logp)  # Probability ratio
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantage
policy_loss = -torch.min(surr1, surr2).mean()  # Clipped
```

### GAE:
```python
delta = reward + gamma * next_value - value  # TD error
advantage = delta + gamma * lambda * advantage  # Recursive
```

### Total Loss:
```python
loss = policy_loss + coef_v * value_loss - coef_ent * entropy
```

---

**NOT:** Bu doküman özet notlar içindir. Detaylı kod açıklamaları için `lunar_lander_ppo_gym.py` ve `PPO_ANALYSIS_TR.md` dosyalarına bakın.
