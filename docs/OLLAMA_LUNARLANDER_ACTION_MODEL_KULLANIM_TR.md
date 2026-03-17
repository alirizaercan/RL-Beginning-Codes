# LunarLander Action Model Ollama Kullanim Rehberi

Bu dokuman, `qwen2.5:0.5b` tabanli fine-tune edilmis LunarLander action modelini iki farkli yolla nasil kullanacagini anlatir:

1. GGUF dosyasi + `Modelfile` ile lokal Ollama modeli olusturmak
2. Publish edilmis Ollama modeli varsa dogrudan onu cekip kullanmak

Bu modelin gorevi:
- girdi olarak LunarLander state almak
- cikti olarak action tahmini vermek

Bu model bir reward modeli degildir.

## 0. Model Linkleri

Bu proje icin olusturdugumuz modeller:

- LoRA modeli: `https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-lora`
- GGUF modeli: `https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-gguf`

Bu linklerin anlami:

- `LoRA` linki:
  - fine-tune adapter dosyalarini icerir
  - daha cok Hugging Face / Transformers tarafinda kullanmak icin uygundur
  - dogrudan Ollama ile kullanilmaz

- `GGUF` linki:
  - Ollama / llama.cpp uyumlu dosyalari icerir
  - arkadasinin lokal Ollama modeli olusturmasi icin esas kullanacagi link budur

Arkadasin amaci Ollama ile modeli calistirmaksa once `GGUF` linkine bakmasi gerekir.

## 1. Gereksinimler

Bilgisayarinda su kurulu olmali:

- `ollama`
- Python kullaniyorsan `ollama` Python paketi

Python paketi kurmak icin:

```bash
pip install ollama
```

## 2. Yontem A: GGUF Dosyasindan Lokal Ollama Modeli Olusturma

Bu yontemde `.gguf` dosyasini indirip kendi bilgisayarinda Ollama modeli olusturursun.

GGUF dosyasini su repodan indirebilirsin:

- `https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-gguf`

### 2.1. GGUF dosyasini hazirla

Elinde buna benzer bir dosya olmali:

```text
qwen2.5-0.5b.Q4_K_M.gguf
```

Bu dosyayi GGUF reposundaki `Files and versions` bolumunden indirebilirsin.

Ornek olarak bir klasor olustur:

```bash
mkdir -p ~/ollama-lunarlander
cd ~/ollama-lunarlander
```

Sonra `.gguf` dosyasini bu klasore koy.

### 2.2. `Modelfile` olustur

Terminalde su komutu calistir:

```bash
printf 'FROM ./qwen2.5-0.5b.Q4_K_M.gguf\n' > Modelfile
```

Istersen dosyayi kontrol et:

```bash
cat Modelfile
```

Beklenen icerik:

```text
FROM ./qwen2.5-0.5b.Q4_K_M.gguf
```

Bu en basit `Modelfile` ornegidir. Yani sadece GGUF dosyasini baz alir.

Istersen `Modelfile` dosyasini ozellestirebilirsin. Ornegin:

- modele sabit bir sistem mesaji ekleyebilirsin
- sicaklik gibi generation parametrelerini ayarlayabilirsin
- cevabi daha kisa veya daha kontrollu hale getirmeyi deneyebilirsin

Ornek gelismis `Modelfile`:

```text
FROM ./qwen2.5-0.5b.Q4_K_M.gguf

SYSTEM """
You are a LunarLander action prediction model.
Given a LunarLander state, return the next action.
Keep the answer short and focused.
"""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_predict 32
```

Bu alanlar ne ise yarar:

- `SYSTEM`:
  - modele kalici bir davranis talimati verir
  - her promptta ayni kurali tekrar yazmak zorunda kalmazsin

- `PARAMETER temperature 0.2`:
  - cevabi daha deterministik yapar
  - action prediction gibi gorevlerde genelde dusuk temperature daha uygundur

- `PARAMETER top_p 0.9`:
  - token secim uzayini kontrol eder
  - genelde varsayilan seviyelerde birakilabilir

- `PARAMETER num_predict 32`:
  - maksimum cikti uzunlugunu sinirlar
  - kisa action cevabi icin faydalidir

Modelfile'i degistirdikten sonra modeli yeniden olusturman gerekir:

```bash
ollama create lunarlander-action -f Modelfile
```

Eger daha once ayni isimle model olusturduysan, bu komut modeli yeni `Modelfile` ayarlariyla gunceller.

### 2.3. Ollama modelini olustur

```bash
ollama create lunarlander-action -f Modelfile
```

Basarili olursa artik lokal model adin su olur:

```text
lunarlander-action
```

### 2.4. Modeli test et

```bash
ollama run lunarlander-action
```

Direkt prompt ile test etmek istersen:

```bash
ollama run lunarlander-action "State: [x=0.0005, y=1.4126, vx=0.0492, vy=0.0739, angle=-0.0006, angular_vel=-0.0112, left_leg=0.0000, right_leg=0.0000]. What action should the lander take?"
```

## 3. Yontem B: Publish Edilmis Ollama Modelini Dogrudan Kullanma

Eger model Ollama uzerinde publish edildiyse, `.gguf` ile ugrasmadan dogrudan cekebilirsin.

Ornek model adi:

```text
alirizaercan1/lunarlander-action
```

### 3.1. Modeli cek

```bash
ollama pull alirizaercan1/lunarlander-action
```

Alternatif olarak:

```bash
ollama run alirizaercan1/lunarlander-action
```

Bu komut modeli yoksa once indirir, sonra calistirir.

### 3.2. Test et

```bash
ollama run alirizaercan1/lunarlander-action "State: [x=0.0005, y=1.4126, vx=0.0492, vy=0.0739, angle=-0.0006, angular_vel=-0.0112, left_leg=0.0000, right_leg=0.0000]. What action should the lander take?"
```

## 4. Python Kodunda Kullanma

Eger daha once kodda normal Qwen modeli kullaniyorsan, sadece `model=` degerini degistirmen yeterli.

### 4.1. Onceki kullanim

```python
import ollama

response = ollama.chat(
    model="qwen2.5:0.5b",
    messages=[
        {"role": "user", "content": "State: [...] What action should the lander take?"}
    ],
)

print(response["message"]["content"])
```

### 4.2. Yeni kullanim

Publish edilmis modeli kullaniyorsan:

```python
import ollama

response = ollama.chat(
    model="alirizaercan1/lunarlander-action",
    messages=[
        {"role": "user", "content": "State: [...] What action should the lander take?"}
    ],
)

print(response["message"]["content"])
```

Lokal GGUF'den olusturdugun modeli kullaniyorsan:

```python
import ollama

response = ollama.chat(
    model="lunarlander-action",
    messages=[
        {"role": "user", "content": "State: [...] What action should the lander take?"}
    ],
)

print(response["message"]["content"])
```

## 5. Kisa Ozet

Iki kullanim sekli var:

1. GGUF dosyasindan lokal model olustur:
   - `Modelfile`
   - `ollama create lunarlander-action -f Modelfile`
   - Python'da `model="lunarlander-action"`

2. Publish edilmis modeli dogrudan kullan:
   - `ollama pull alirizaercan1/lunarlander-action`
   - Python'da `model="alirizaercan1/lunarlander-action"`

3. Hugging Face tarafinda modelleri incelemek istersen:
   - LoRA: `https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-lora`
   - GGUF: `https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-gguf`

## 6. Notlar

- Bu model action prediction icindir.
- Reward uretmek icin ayri bir reward modeli gerekir.
- Cikti bazen dogal dilde olabilir; gerekirse prompt daha kisitlayici yazilabilir.
- Ollama lokal kullaniminda ek auth gerekmez.
- Publish edilmis modeli cekmek veya publish etmek icin Ollama hesabiyla giris gerekebilir.
- LoRA modeli ile GGUF modeli ayni amaca hizmet etse de farkli artifact tipleridir.
- Arkadasin sadece Ollama kullanacaksa esasen `GGUF` modeli yeterlidir.
