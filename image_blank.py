import os
from PIL import Image
import numpy as np

# Função para verificar se uma imagem é branca
def is_image_white(image_path, threshold=240):
    # Abrir a imagem
    img = Image.open(image_path)
    
    # Converter para RGB (caso não seja)
    img = img.convert('RGB')

    # Converter a imagem para uma matriz NumPy
    img_array = np.array(img)
    
    # Verificar se todos os pixels são brancos ou muito próximos de brancos
    # Um pixel é considerado branco se todos os valores R, G e B estiverem acima do threshold
    white_pixels = np.all(img_array >= threshold, axis=-1)
    
    # Se a maioria dos pixels for branca, a imagem é considerada branca
    return np.all(white_pixels)

# Caminho da pasta com as imagens
image_folder_path = "/home/mrodriguesoliv/datasetcolpaliimage"

# Iterar sobre todos os arquivos da pasta de imagens
for filename in os.listdir(image_folder_path):
    # Verificar se o arquivo é uma imagem (com as extensões .jpg, .jpeg ou .png)
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, filename)
        
        # Verificar se a imagem é branca
        if is_image_white(image_path):
            try:
                # Deletar a imagem se for branca
                os.remove(image_path)
                print(f"Imagem {filename} deletada, pois é branca.")
            except Exception as e:
                print(f"Erro ao deletar a imagem {filename}: {e}")
        else:
            print(f"A imagem {filename} não é branca.")
