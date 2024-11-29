import base64
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os
from datasets import Dataset
import json


load_dotenv(dotenv_path='.env-local')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
image_folder_path = "/home/mrodriguesoliv/datasetcolpaliimage"

# Lista para armazenar os dados gerados
data = []

for filename in os.listdir(image_folder_path):

    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, filename)
        
        base64_image = encode_image(image_path)

        try:
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[  
                {
                "role": "system",
                "content": "You are a helpful assistant that can interpret images and answer questions about them. If the image contains text, try to summarize it. If it contains objects or scenes, describe them in detail."
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "You are an assistant specialized in Multimodal RAG tasks. The task is the following: given an image from a pdf page, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus. The question should be relevant to the page, and should not be too specific or too general. The question should be about the subject of the page, and the answer needs to be found in the page. Remember that the question is asked by a user to get some information from a large documentary corpus that contains multimodal data. Generate a question that could be asked by a user without knowing the existence and the content of the corpus. Generate as well the answer to the question, which should be found in the page. And the format of the answer should be a list of words answering the question.Generate at most THREE pairs of questions and answers per page as text. return a structured json, where each question is passed as question_x where x is the question number and the answer is passed as answer_x where x is the answer number, DO NOT RETURN ANYTHING OTHER THAN JSON. DO NOT RETURN ```JSON AT START. ANSWERS SHOULD NOT BE A LIST, THEY SHOULD BE A PHRASE, SENTENCE, ALL QUESTIONS AND ANSWER IN PORTUGUESE."
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

            gpt_response = response.choices[0].message.content

            print(f'RESPOSTA: {gpt_response}')

            gpt_response_json = json.loads(gpt_response)

            data.append({
                "INPUT_GPT_1": gpt_response_json.get("question_1",""),
                "OUTPUT_GPT_1": gpt_response_json.get("answer_1",""),
                "INPUT_GPT_2": gpt_response_json.get("question_2",""),
                "OUTPUT_GPT_2": gpt_response_json.get("answer_2",""),
                "INPUT_GPT_3": gpt_response_json.get("question_3",""),
                "OUTPUT_GPT_3": gpt_response_json.get("answer_3","")
            })
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

dataset = Dataset.from_dict({
    "INPUT_GPT_1": [item["INPUT_GPT_1"] for item in data],
    "OUTPUT_GPT_1": [item["OUTPUT_GPT_1"] for item in data],
    "INPUT_GPT_2": [item["INPUT_GPT_2"] for item in data],
    "OUTPUT_GPT_2": [item["OUTPUT_GPT_2"] for item in data],
    "INPUT_GPT_3": [item["INPUT_GPT_3"] for item in data],
    "OUTPUT_GPT_3": [item["OUTPUT_GPT_3"] for item in data],
})            

dataset.push_to_hub("gpt4v-briefings")

print("Dados enviados para o Hugging Face com sucesso!")