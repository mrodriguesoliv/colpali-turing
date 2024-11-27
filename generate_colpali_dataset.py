import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import login
from collections import defaultdict
from pydantic import BaseModel, ValidationError

# Habilitar a transferência do Hub do Hugging Face
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Carregar variáveis de ambiente
load_dotenv(dotenv_path='.env-local')

# Configuração de autenticação
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
hub_token = os.getenv("HUGGING_FACE_TOKEN")

print(f'testeee: {OPENAI_API_KEY}')

# Login no Hugging Face
login(token=hub_token)
openai.api_key = OPENAI_API_KEY

# Definindo um modelo de saída para validação
class GPTOutput(BaseModel):
    input: str
    output: str

# Carregar dataset
dataset = load_dataset("mrodriguesoliv/colpali-turing")

# Lista para armazenar os resultados
briefings = []

print(dataset.keys())

client = OpenAI()

for row in tqdm(dataset["train"], desc="Processando imagens"):
    try:
        image_path = row["image"]

        prompt = "Baseando-se na análise da imagem, descreva os principais elementos visuais e textuais em detalhes. Gere uma pergunta e uma resposta contextual para treinar modelos multimodais."

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente que analisa imagens e gera inputs e outputs úteis para treinamento multimodal."},
                {"role": "user", "content": prompt}
            ],
        )

        gpt_response = response["choices"][0]["message"]["content"]

        validated_output = GPTOutput(
            input=prompt,
            output=gpt_response
        )

        # Adicionar resultados à lista
        briefings.append({
            "image_path": image_path,
            "input": validated_output.input,
            "output": validated_output.output
        })

    except ValidationError as ve:
        # Caso ocorra um erro de validação, adicionar o erro nos resultados
        briefings.append({
            "image_path": image_path,
            "input": "Deu erro",
            "output": None,
            "error": str(ve)
        })
    except Exception as e:
        # Caso ocorra qualquer outro erro, adicionar o erro nos resultados
        briefings.append({
            "image_path": image_path,
            "input": "Deu erro",
            "output": None,
            "error": str(e)
        })  

# Organizar os dados para criar o dataset final
columns = defaultdict(list)
for briefing in briefings:
    for key, value in briefing.items():
        columns[key].append(value)

# Criar dataset e enviar para o Hugging Face Hub
result_dataset = Dataset.from_dict(columns)
result_dataset.push_to_hub("colpali_gpt4v_briefings")
