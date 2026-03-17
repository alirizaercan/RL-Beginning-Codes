# Hugging Face ve Ollama Hesabi Acma, Model Kaydetme ve Yukleme

Bu rehber, Qwen2.5-0.5B modelini fine-tune ettikten sonra:

1. modeli kaydetme,
2. Colab'dan kendi bilgisayarina indirme,
3. Hugging Face Hub'a yukleme,
4. GGUF olarak kaydetme,
5. Ollama'ya ekleme ve istersek ollama.com hesabina publish etme

akisini sifirdan anlatir.

## Kisa cevap

Evet, bu akisin tamami mumkun:

1. Colab'da modeli veya LoRA adapter'i kaydedebilirsin.
2. Dosyalari bilgisayarina indirebilirsin.
3. Hugging Face hesabina model olarak yukleyebilirsin.
4. GGUF dosyasi olusturabilirsen bunu Ollama'ya local model olarak ekleyebilirsin.
5. Ollama hesabin varsa modeli publish etmeyi de deneyebilirsin.

Ama burada iki ayri format oldugunu net ayirmak gerekir:

1. Hugging Face tarafinda:
   - genelde model klasoru
   - LoRA adapter
   - tokenizer
   - config dosyalari

2. Ollama tarafinda:
   - genelde GGUF veya Modelfile tabanli model

Yani Hugging Face'e yukledigin klasoru dogrudan Ollama'ya atmazsin. Gerekirse once GGUF uretirsin veya Ollama'nin destekledigi formata cevirirsin.

## 1. Hugging Face hesabi nasil acilir

Resmi kaynaklar:
- HF signup: https://huggingface.co/join
- HF repo baslangic: https://huggingface.co/docs/hub/en/repositories-getting-started
- HF model upload docs: https://huggingface.co/docs/hub/en/models-uploading
- HF CLI docs: https://huggingface.co/docs/huggingface_hub/en/guides/cli

Adimlar:

1. `https://huggingface.co/join` sayfasina git.
2. Kullanici adi, e-posta ve sifre ile hesap ac.
3. E-posta dogrulamasini yap.
4. Giris yaptiktan sonra sag ustten `Settings` kismina git.
5. `Access Tokens` bolumunden bir token olustur.
6. Baslangic icin `Write` yetkili token yeterlidir.

Token olusturduktan sonra Colab veya bilgisayarda giris icin:

```bash
huggingface-cli login
```

ve token'i yapistirirsin.

Alternatif:

```python
from huggingface_hub import login
login("hf_your_token_here")
```

## 2. Ollama hesabi nasil acilir

Resmi kaynaklar:
- Ollama sign up: https://ollama.com/signup
- Ollama cloud docs: https://docs.ollama.com/cloud
- Ollama auth docs: https://docs.ollama.com/api/authentication
- Ollama push docs: https://docs.ollama.com/api/push

Adimlar:

1. `https://ollama.com/signup` adresine git.
2. Hesap olustur.
3. Sonra kendi bilgisayarinda terminalde:

```bash
ollama signin
```

4. Acilan giris akisindan hesabini bagla.

Not:
- Lokal `localhost` uzerinden calisan Ollama API'si icin auth gerekmez.
- Ama cloud model kullanmak, private model cekmek veya model publish etmek icin auth gerekir.

## 3. Colab'da modeli kaydetme

Senin mevcut notebook ve script akisin LoRA adapter olarak kaydediyor:

```python
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

Bu genelde su tip dosyalar uretir:
- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer dosyalari
- ek config dosyalari

Bu dosyalar tam merge edilmis full model olmayabilir; cogu zaman LoRA adapter olur.

## 4. Colab'dan bilgisayara indirme

Colab'da klasoru zipleyip indirebilirsin:

```python
!zip -r qwen25_05b_lora.zip qwen25_05b_lunarlander_action_lora
from google.colab import files
files.download("qwen25_05b_lora.zip")
```

Alternatif olarak Google Drive'a da kaydedebilirsin.

## 5. Hugging Face Hub'a model yukleme

### Yontem 1: Web arayuzu

Resmi dokumana gore HF model repo'su web arayuzunden de yuklenebilir.

Adimlar:

1. `https://huggingface.co/new` sayfasina git.
2. `Model` repo olustur.
3. Repo adini belirle.
4. `Files and versions` tabina gir.
5. `Add file -> Upload files` ile klasordeki dosyalari yukle.

Bu yontem kucuk dosyalar icin basit ama buyuk dosyalarda CLI daha saglikli.

### Yontem 2: Python ile push

Eger model / tokenizer objesi elindeyse:

```python
model.push_to_hub("kullanici_adi/repo_adi")
tokenizer.push_to_hub("kullanici_adi/repo_adi")
```

Ama LoRA adapter veya local klasor yuklemek icin pratik yol genelde CLI veya `upload_folder`.

### Yontem 3: huggingface_hub ile klasor yukleme

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="kullanici_adi/qwen25-05b-lunarlander", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="qwen25_05b_lunarlander_action_lora",
    repo_id="kullanici_adi/qwen25-05b-lunarlander",
    repo_type="model",
)
```

## 6. GGUF olarak kaydetme

Eger notebook veya kullandigin framework GGUF export destekliyorsa, bunu ek adim olarak yapabilirsin.

Unsloth notebook'larda genelde buna benzer hucreler olur:

```python
model.save_pretrained_gguf("model", tokenizer)
```

veya belirli quantization secenekleri:

```python
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
```

Burada dikkat:
- Her notebook / her model / her adapter akisi GGUF export'u ayni sekilde desteklemeyebilir.
- LoRA adapter'i bazen once merge etmek gerekebilir.

## 7. GGUF dosyasini Ollama'ya ekleme

Ollama tarafinda resmi yontem `Modelfile` ile import etmektir.

Ollama dokumanlarinda yer alan mantik:

1. Bir `Modelfile` olustur.
2. GGUF dosyasina `FROM` ile referans ver.
3. `ollama create` ile modeli kaydet.

Ornek:

`Modelfile`

```text
FROM ./qwen25_lunarlander.gguf
```

Komut:

```bash
ollama create lunarlander-qwen -f Modelfile
ollama run lunarlander-qwen
```

Bu adimdan sonra model artik kendi bilgisayarinda Ollama icinde calisir.

## 8. Ollama hesabina model publish etme

Resmi dokumanlarda publish/push icin API ve auth anlatiliyor.

Auth:

```bash
ollama signin
```

Push mantigi:

```bash
curl http://localhost:11434/api/push -d '{
  "model": "kullanici_adi/model_adi"
}'
```

Pratikte CLI veya lokal Ollama instance uzerinden de namespace ile push mantigi kullanilir.

Onemli not:
- Ollama tarafinda publish akisi Hugging Face kadar yaygin ve esnek olmayabilir.
- Once local modelin sorunsuz olustugundan emin ol.
- Hesapla giris yapmadan push deneme.

## 9. Bu proje icin onerilen akisi

En temiz sira su:

1. Colab'da Qwen2.5-0.5B fine-tune et.
2. LoRA adapter'i lokal klasore kaydet.
3. Klasoru zipleyip bilgisayarina indir.
4. Hugging Face hesabina yukle.
5. Gerekirse merge / GGUF export yap.
6. GGUF dosyasini Ollama'ya local import et.
7. Sonra istersen Ollama hesabina publish etmeyi dene.

Bu sira neden daha iyi?

1. Hugging Face tarafi model saklama ve paylasma icin daha standart.
2. Ollama tarafi deployment / local run icin iyi.
3. Once HF tarafini garantiye almak daha guvenli.

## 10. Hangi dosya nereye gider

### Hugging Face icin

Yuklemek isteyecegin tipik dosyalar:
- LoRA adapter dosyalari
- tokenizer dosyalari
- config dosyalari
- README / model card

### Ollama icin

Yuklemek isteyecegin sey:
- GGUF model dosyasi
- veya Modelfile ile referans verilen bir local model dosyasi

## 11. Kisa ozet

Evet, senin planin teknik olarak yapilabilir:

1. Fine-tuned modeli kaydet
2. PC'ye indir
3. Hugging Face Hub'a yukle
4. GGUF olarak export et
5. Ollama'ya import et
6. Hesabin varsa publish de et

Ama format farkini unutma:
- HF = repo / adapter / tokenizer / config
- Ollama = GGUF + Modelfile mantigi

## 12. Yararlı resmi linkler

Hugging Face:
- https://huggingface.co/join
- https://huggingface.co/docs/hub/en/models-uploading
- https://huggingface.co/docs/hub/en/repositories-getting-started
- https://huggingface.co/docs/huggingface_hub/en/guides/cli

Ollama:
- https://ollama.com/signup
- https://docs.ollama.com/cloud
- https://docs.ollama.com/api/authentication
- https://docs.ollama.com/api/push
