import os
from dotenv import load_dotenv
import openai
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import login
from collections import defaultdict
from pydantic import BaseModel, ValidationError
from PIL import Image
import io

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
hub_token = os.getenv("HUGGING_FACE_TOKEN")

login(token=hub_token)
openai.api_key = OPENAI_API_KEY

class GPTOutput(BaseModel):
    input: str
    output: str

dataset = load_dataset("mrodriguesoliv/colpali-turing")

briefings = []

def pil_to_bytes(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()

for row in tqdm(dataset["train"], desc="Processando imagens"):
    try:
        
        image_path = row["image"]
        
        image = Image.open(image_path)
        
        image_data = pil_to_bytes(image)

        prompt = "Baseando-se na análise da imagem, descreva os principais elementos visuais e textuais em detalhes. Gere uma pergunta e uma resposta contextual para treinar modelos multimodais."
    
        response = openai.ChatCompletion.create(
            model="gpt-4-vision",
            messages=[
                {"role": "system", "content": "Você é um assistente que analisa imagens e gera inputs e outputs úteis para treinamento multimodal."},
                {"role": "user", "content": prompt}
            ],
            files=[{"file": ("imagem.jpg", image_data)}]
        )

        gpt_response = response["choices"][0]["message"]["content"]

        validated_output = GPTOutput(
            input=prompt,
            output=gpt_response
        )

        briefings.append({
            "image_path": image_path,
            "input": validated_output.input,
            "output": validated_output.output
        })

    except ValidationError as ve:
        briefings.append({
            "image_path": image_path,
            "input": None,
            "output": None,
            "error": str(ve)
        })
    except Exception as e:
        briefings.append({
            "image_path": image_path,
            "input": None,
            "output": None,
            "error": str(e)
        })

columns = defaultdict(list)
for briefing in briefings:
    for key, value in briefing.items():
        columns[key].append(value)

result_dataset = Dataset.from_dict(columns)

result_dataset.push_to_hub("colpali_gpt4v_briefings")
