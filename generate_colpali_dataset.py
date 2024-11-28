import base64
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path='.env-local')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
image_folder_path = "/home/mrodriguesoliv/datasetcolpaliimage"

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
                    "text": "You are an assistant specialized in Multimodal RAG tasks. The task is the following: given an image from a pdf page, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus. The question should be relevant to the page, and should not be too specific or too general. The question should be about the subject of the page, and the answer needs to be found in the page. Remember that the question is asked by a user to get some information from a large documentary corpus that contains multimodal data. Generate a question that could be asked by a user without knowing the existence and the content of the corpus. Generate as well the answer to the question, which should be found in the page. And the format of the answer should be a list of words answering the question.Generate at most THREE pairs of questions and answers per page as JSON with the following format, answer ONLY using JSON, NOTHING ELSE. Example:",
                    "type": "text",
                    "example": {
                        "description": "Where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers that could be as long as needed.",
                        "questions": [
                            {
                                "question": "XXXXXX",
                                "answer": ["YYYYYY"]
                            },
                            {
                                "question": "XXXXXX",
                                "answer": ["YYYYYY"]
                            },
                            {
                                "question": "XXXXXX",
                                "answer": ["YYYYYY"]
                            }
                        ]
                    }
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

            print(f'REPOSTA ESTÁ AQUI:')
            print(response.choices[0])

        except Exception as e:
            print(f"Error processing {filename}: {e}")