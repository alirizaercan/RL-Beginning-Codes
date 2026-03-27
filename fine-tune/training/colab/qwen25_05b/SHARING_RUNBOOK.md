# Sharing The Final Model

Final model directory:

`/home/autodidactic/Desktop/qwen25_fullft_run/LLaMA-Factory/saves/qwen25_05b_base_full_ft_lunarlander_a4000`

## Recommended: Hugging Face

This model is already saved in Transformers/Safetensors format, so the cleanest path is to upload the final output directory directly.

### 1. Create a zip archive

Run inside `LLaMA-Factory/`:

```bash
python archive_finetuned_model.py \
  --output-dir saves/qwen25_05b_base_full_ft_lunarlander_a4000 \
  --archive-base qwen25_05b_base_full_ft_lunarlander_a4000
```

### 2. Log in to Hugging Face

```bash
hf auth login
```

### 3. Create a model repo

Replace `YOUR_USERNAME` with the target account or org name:

```bash
hf repo create YOUR_USERNAME/qwen25_05b_base_full_ft_lunarlander_a4000 --private
```

### 4. Upload the folder

```bash
hf upload YOUR_USERNAME/qwen25_05b_base_full_ft_lunarlander_a4000 \
  /home/autodidactic/Desktop/qwen25_fullft_run/LLaMA-Factory/saves/qwen25_05b_base_full_ft_lunarlander_a4000 \
  .
```

## Ollama

The official Ollama import docs clearly document Safetensors model import for Llama, Mistral, Gemma, and Phi3 architectures. They do not explicitly list Qwen for direct Safetensors import.

Because of that, the safest Ollama path for this Qwen full fine-tuned model is:

1. keep the canonical model on Hugging Face
2. convert it to GGUF if Ollama delivery is required
3. build an Ollama model from the GGUF file

### Modelfile example for a GGUF export

```text
FROM ./qwen25_05b_base_full_ft_lunarlander_a4000.gguf
PARAMETER temperature 0
PARAMETER num_ctx 2048
SYSTEM You predict the next LunarLander action. Output exactly one line in the format: Action: <0|1|2|3>
```

### Build and run

```bash
ollama create lunarlander-qwen25 -f Modelfile
ollama run lunarlander-qwen25
```

### Share on ollama.com

```bash
ollama cp lunarlander-qwen25 YOUR_OLLAMA_USERNAME/lunarlander-qwen25
ollama push YOUR_OLLAMA_USERNAME/lunarlander-qwen25
```

## Recommendation

For this project:

- use Hugging Face as the primary artifact store and sharing endpoint
- use Ollama only as a secondary packaging target if the receiving side specifically wants `ollama run ...`
