# Bipedal Walker PPO + LLM Reward Dokumani (XYZ - Junior Engineer)

Bu dokuman iki dosyayi birlikte aciklar:

1. `bipedal-walker/bipedal_walker_ppo_minimal.py`
2. `bipedal-walker/bipedal_walker_llm_custom_env.py`

Amac: Kodlari satir mantigiyla anlayip Team Lead sorularina teknik ve net cevap verebilmek.

---

## X = Ne Problemi Cozuyoruz?

BipedalWalker icin iki farkli yaklasim kuruyoruz:

1. **Klasik PPO egitimi**
- Reward dogrudan environment tarafindan verilir.
- Ajan bu reward ile policy ogrenir.

2. **Custom LLM reward environment + PPO**
- Transition (state/action/termination) environment'dan gelir.
- Reward environment'dan **ignore** edilir.
- Reward her step LLM (Ollama / qwen2.5:7b) ile hesaplanir.
- PPO bu yeni reward sinyaliyla ogrenmeye calisir.

Kisa fark:
- Dosya 1: "normal RL"
- Dosya 2: "reward replacement RL"

---

## Y = Neden Bu Mimariyi Sectik?

### Neden `bipedal_walker_ppo_minimal.py` var?
- Baseline almak icin.
- "Sistem normalde nasil calisiyor" sorusuna net cevap vermek icin.
- LLM reward yaklasiminin etkisini karsilastirmak icin referans nokta.

### Neden `bipedal_walker_llm_custom_env.py` var?
- Hoca beklentisi: reward'u dis kaynaktan (LLM) belirlemek.
- Wrapper yerine sifirdan custom `gym.Env` sinifi ile acik kontrol gostermek.

### Neden PPO her iki senaryoda da var?
- Ayni algoritma ile sadece reward kanalini degistirip farki gozlemlemek icin.

---

## Z = Kodlar Nasil Calisiyor?

# 1) `bipedal_walker_ppo_minimal.py` (Klasik PPO)

## 1.1 Importlar

```python
import gymnasium as gym
from stable_baselines3 import PPO
```

- `gymnasium`: environment
- `PPO`: policy gradient tabanli RL algoritmasi (Stable-Baselines3)

## 1.2 Training env olusturma

```python
train_env = gym.make("BipedalWalker-v3")
```

- Render kapali (hizli egitim)
- `action_space` ve `observation_space` print'leri debug icin

## 1.3 Model olusturma

```python
model = PPO("MlpPolicy", train_env, verbose=1)
```

- `MlpPolicy`: standart fully-connected policy network
- `verbose=1`: egitim loglarini yazdirir

## 1.4 Egitim

```python
TOTAL_TIMESTEPS = 50_000
model.learn(total_timesteps=TOTAL_TIMESTEPS)
```

- 50k demo icin orta seviye bir deger
- Arttirirsan davranis genelde iyilesir, sure uzar

## 1.5 Evaluation (render acik)

```python
env = gym.make("BipedalWalker-v3", render_mode="human")
```

- Burada model predict eder:

```python
action, _state = model.predict(observation, deterministic=True)
```

- `deterministic=True`: testte daha stabil aksiyon secimi

## 1.6 Episode dongusu

```python
observation, reward, terminated, truncated, info = env.step(action)
```

- Bu dosyada reward klasik env reward'dur.
- `episode_over = terminated or truncated` ile dongu bitirilir.

---

# 2) `bipedal_walker_llm_custom_env.py` (Custom Env + LLM Reward)

## 2.1 Importlar

```python
import re
import gymnasium as gym
import ollama
from stable_baselines3 import PPO
```

- `re`: LLM cevabindan sayi parse etmek icin
- `ollama`: local model cagrisi

## 2.2 Custom environment sinifi

```python
class CustomBipedalWalkerLLMEnv(gym.Env):
```

Bu sinif **wrapper degil**, sifirdan yazilmis bir `gym.Env`.

### Ic mantik
- Icinde bir `source_env` tutuyor:

```python
self.source_env = gym.make("BipedalWalker-v3", render_mode=render_mode)
```

- State/action transition buradan geliyor.
- Ama reward bu env'den alinmiyor.

## 2.3 Space uyumlulugu

```python
self.action_space = self.source_env.action_space
self.observation_space = self.source_env.observation_space
```

PPO'nun custom env'i sorunsuz kullanmasi icin zorunlu.

## 2.4 `reset()`

```python
observation, info = self.source_env.reset(**kwargs)
self.step_count = 0
self.total_llm_reward = 0.0
self.prev_state = observation
```

- Episode sayaçlari sifirlanir
- Ilk state saklanir

## 2.5 `step(action)` cekirdek akis

```python
next_state, _env_reward, terminated, truncated, info = self.source_env.step(action)
```

- `_env_reward` alinip **kullanilmiyor**

```python
llm_reward, llm_text = self._compute_llm_step_reward(...)
```

- Her step reward LLM'den istenir
- Sonra `info` icine debug bilgisi konur

## 2.6 LLM reward fonksiyonu

```python
def _compute_llm_step_reward(...):
```

Bu fonksiyon:
1. State'i isimli dict'e cevirir (`_state_to_named_dict`)
2. Action'i isimli dict'e cevirir (`_action_to_named_dict`)
3. Detayli reward prompt'u kurar
4. `ollama.chat(..., temperature=0)` ile model cagirir
5. Cevabi float parse eder
6. `[-100, 100]` araligina clamp eder

### Neden named dict?

Ham liste yerine su tipte gonderim yorumlamayi kolaylastirir:

```python
{
  "hull_angle": ...,
  "hull_velocity_x": ...,
  ...
}
```

Bu, prompt anlasilirligini artirir.

## 2.7 Parse guvenligi

```python
def _safe_parse_float(text):
```

- Direkt `float(text)` dener
- Olmazsa regex ile ilk sayiyi bulur
- Hala olmazsa `0.0`

Amaç: LLM cevabi her zaman temiz sayi olmayabilir; kodu patlatmamak.

## 2.8 PPO ile custom env egitimi

```python
train_env = CustomBipedalWalkerLLMEnv(model_name="qwen2.5:7b", render_mode=None)
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=5_000)
```

- LLM her step cagrildigi icin egitim yavas olabilir
- O yüzden timesteps dusuk tutulmus

## 2.9 PPO ile test

```python
env = CustomBipedalWalkerLLMEnv(model_name="qwen2.5:7b", render_mode="human")
action, _ = model.predict(observation, deterministic=True)
```

- Burada reward env'den degil LLM'den gelir

---

## Kritik Tasarim Kararlari

1. `temperature=0`
- Benzer girdide benzer cikti hedeflenir.
- Test repeatability artar.

2. Her step LLM reward
- Arastirma acisindan esnek
- Pratikte gecikme ve maliyet yuksek

3. Clamp (`[-100, 100]`)
- LLM asin sayilar donerse egitimi bozmasin

4. `info` icine `llm_text` koyma
- Debug ve Team Lead incelemesinde buyuk kolaylik

---

## Team Lead Gelebilecek Sorular ve Hazir Cevaplar

### Soru 1: Bu custom env wrapper mi?
Cevap: Hayir. `gym.Env`'den turetilmis sifirdan bir environment. Wrapper kullanilmiyor.

### Soru 2: Transition kimden geliyor?
Cevap: `source_env = gym.make("BipedalWalker-v3")` icinden geliyor.

### Soru 3: Env reward kullaniyor musunuz?
Cevap: Hayir. `_env_reward` aliniyor ama ignore ediliyor. Reward LLM'den geliyor.

### Soru 4: PPO buna nasil uyum sagliyor?
Cevap: `action_space` ve `observation_space` birebir source_env ile uyumlu oldugu icin PPO custom env'i normal Gym env gibi kullanabiliyor.

### Soru 5: LLM hep benzer sayilar donuyor, neden?
Cevap: `temperature=0` + promptun benzer patternleri modelin benzer skorlar vermesine yol aciyor. Parse islemi de ilk sayiyi aliyor.

### Soru 6: Bu production-ready mi?
Cevap: Hayir. Bu bir arastirma/demo prototipi. Her step LLM cagri maliyetli ve yavas.

### Soru 7: Bu egitim stabil mi?
Cevap: LLM reward non-stationary olabilecegi icin klasik env reward kadar stabil degil.

---

## Pratik Iyilestirme Onerileri

1. Promptu daha formullu yap:
- Ornek: `reward = 20*vx - 5*abs(angle) - 2*mean_abs_action`

2. LLM'i her step yerine her N step / episode sonu cagir.

3. `llm_text` + state/action logunu dosyaya yaz.

4. Ikinci bir fallback reward formulu ekle:
- LLM hata verirse deterministic formula don.

---

## Kisa Ozet

- `bipedal_walker_ppo_minimal.py`: klasik PPO baseline
- `bipedal_walker_llm_custom_env.py`: custom env + LLM step reward + PPO
- En kritik fark: reward kaynagi
  - Baseline: environment
  - Custom: LLM

Bu iki dosya birlikte, "reward replacement" konseptini teknik olarak gostermek icin dogru bir cift olusturur.
