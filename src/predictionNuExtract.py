import json
import os
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suprime avisos sobre flash-attention se não estiver disponível
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*flash-attention.*")

def predict_NuExtract(model, tokenizer, texts, template, batch_size=1, max_length=10_000, max_new_tokens=4_000):
    template = json.dumps(json.loads(template), indent=4)
    prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]
    
    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)

            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens, use_cache=False)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return [output.split("<|output|>")[1] for output in outputs]

model_name = "numind/NuExtract-v1.5"

# Diagnóstico detalhado da GPU
print("=== Diagnóstico de GPU ===")
print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA versão: {torch.version.cuda}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memória total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ CUDA não está disponível!")
    print("Possíveis causas:")
    print("  1. PyTorch foi instalado sem suporte CUDA (pip install torch)")
    print("  2. Instale PyTorch com CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  3. Drivers NVIDIA não estão instalados ou atualizados")
    print("  4. GPU não é compatível com CUDA")
print("=" * 30)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"\nUsando device: {device} | dtype: {dtype}")

# Verifica se flash-attention está disponível
try:
    import flash_attn
    flash_attention_available = True
    print("✓ Flash Attention disponível")
except ImportError:
    flash_attention_available = False
    print("⚠ Flash Attention não está instalado (opcional - não afeta funcionalidade)")
    print("  Para instalar: pip install flash-attn --no-build-isolation")
    print("  Nota: Requer CUDA e pode ser complexo no Windows")

print()

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    dtype=dtype, 
    trust_remote_code=True
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Verifica se o modelo realmente está na GPU
if torch.cuda.is_available():
    model_device = next(model.parameters()).device
    print(f"Modelo carregado no device: {model_device}")
    if model_device.type != "cuda":
        print("⚠️ AVISO: Modelo não está na GPU mesmo com CUDA disponível!")

text = """We introduce Mistral 7B, a 7–billion-parameter language model engineered for
superior performance and efficiency. Mistral 7B outperforms the best open 13B
model (Llama 2) across all evaluated benchmarks, and the best released 34B
model (Llama 1) in reasoning, mathematics, and code generation. Our model
leverages grouped-query attention (GQA) for faster inference, coupled with sliding
window attention (SWA) to effectively handle sequences of arbitrary length with a
reduced inference cost. We also provide a model fine-tuned to follow instructions,
Mistral 7B – Instruct, that surpasses Llama 2 13B – chat model both on human and
automated benchmarks. Our models are released under the Apache 2.0 license.
Code: <https://github.com/mistralai/mistral-src>
Webpage: <https://mistral.ai/news/announcing-mistral-7b/>"""

template = """{
    "Model": {
        "Name": "",
        "Number of parameters": "",
        "Number of max token": "",
        "Architecture": []
    },
    "Usage": {
        "Use case": [],
        "Licence": ""
    }
}"""

prediction = predict_NuExtract(model, tokenizer, [text], template)[0]

# Tenta fazer o parse do JSON e formatar a saída
try:
    # Remove possíveis caracteres extras e faz o parse
    prediction_cleaned = prediction.strip()
    # Tenta encontrar o JSON no resultado (caso haja texto antes/depois)
    if '{' in prediction_cleaned:
        json_start = prediction_cleaned.find('{')
        json_end = prediction_cleaned.rfind('}') + 1
        prediction_cleaned = prediction_cleaned[json_start:json_end]
    
    # Faz o parse e formata como JSON indentado
    prediction_json = json.loads(prediction_cleaned)
    print(json.dumps(prediction_json, indent=2, ensure_ascii=False))
except json.JSONDecodeError as e:
    # Se não conseguir fazer o parse, imprime o resultado original
    print("Erro ao fazer parse do JSON. Resultado original:")
    print(prediction)
    print(f"\nErro: {e}")
