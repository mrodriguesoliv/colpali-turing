import base64
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env-local')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if OPENAI_API_KEY is None:
    raise ValueError("A chave da API do OpenAI não foi encontrada no arquivo .env-local")

openai.api_key = OPENAI_API_KEY

client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
image_path = "/home/mrodriguesoliv/datasetcolpaliimage/ManualdoProprierárioP25Metálico_page_15.jpg"

base64_image = encode_image(image_path)

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?",
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}"
          },
        },
      ],
    }
  ],
)

print(response.choices[0])