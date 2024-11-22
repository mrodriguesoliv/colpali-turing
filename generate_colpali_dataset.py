import os
from dotenv import load_dotenv
import openai
from tqdm.auto import tqdm
from datasets import Dataset
from pydantic import BaseModel, ValidationError

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class GPTOutput(BaseModel):
    input: str
    output: str

dataset_path = "mrodriguesoliv/colpali-dataset"

dataset = Dataset.from_huggingface(dataset_path)

briefings = []

for row in tqdm(dataset, desc="Processando imagens"):
    try:
        
        image_path = row["image_path"]
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Prompt para o GPT-4V
        prompt = "Baseando-se na análise da imagem, descreva os principais elementos visuais e textuais em detalhes. Gere uma pergunta e uma resposta contextual para treinar modelos multimodais."
        
        # Enviar para o GPT-4V
        response = openai.ChatCompletion.create(
            model="gpt-4-vision",
            messages=[
                {"role": "system", "content": "Você é um assistente que analisa imagens e gera inputs e outputs úteis para treinamento multimodal."},
                {"role": "user", "content": prompt}
            ],
            files=[
                {"file": (os.path.basename(image_path), image_data)}
            ]
        )
        
        # Obter resposta
        gpt_response = response["choices"][0]["message"]["content"]
        
        # Validar saída com Pydantic
        validated_output = GPTOutput(
            input=prompt,
            output=gpt_response
        )
        
        # Adicionar briefing ao dataset
        briefings.append({
            "image_path": image_path,
            "input": validated_output.input,
            "output": validated_output.output
        })
    
    except ValidationError as ve:
        print(f"Erro de validação na imagem {image_path}: {ve}")
        briefings.append({
            "image_path": image_path,
            "input": None,
            "output": None,
            "error": str(ve)
        })
    except Exception as e:
        print(f"Erro ao processar imagem {image_path}: {e}")
        briefings.append({
            "image_path": image_path,
            "input": None,
            "output": None,
            "error": str(e)
        })

# Criar um novo dataset com os resultados
result_dataset = Dataset.from_dict(briefings)

# Salvar os resultados em disco ou subir no Hugging Face
result_dataset.save_to_disk("colpali_gpt4v_briefings")
