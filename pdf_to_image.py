import os
from pdf2image import convert_from_path

# Caminho para a pasta que contém os arquivos PDF
pdf_folder_path = "/home/mrodriguesoliv/datasetcolpalipdf"
# Caminho para a pasta onde as imagens convertidas serão salvas
image_folder_path = "/home/mrodriguesoliv/datasetcolpaliimage"

# Verificar se a pasta de destino existe, se não, criar
if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)

# Listar todos os arquivos na pasta
for filename in os.listdir(pdf_folder_path):
    # Verificar se o arquivo é um PDF (termina com .pdf)
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, filename)
        
        # Converter todas as páginas do PDF em uma lista de imagens (PIL Image)
        images = convert_from_path(pdf_path, first_page=1, last_page=15)
        
        # Salvar as imagens convertidas em arquivos de imagem
        for i, image in enumerate(images):
            # Criar um nome para a imagem que inclui o nome do PDF e o número da página
            image_filename = f'{os.path.splitext(filename)[0]}_page_{i + 1}.jpg'
            # Salvar a imagem na pasta especificada
            image.save(os.path.join(image_folder_path, image_filename), 'JPEG')
            print(f'Imagem salva: {image_filename}')
