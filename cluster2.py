import cv2
import numpy as np
import os

pasta_dataset = 'dataset'

imagens = []
nomes_arquivos = []

for arquivo in os.listdir(pasta_dataset):
    caminho_completo = os.path.join(pasta_dataset, arquivo)

    if os.path.isfile(caminho_completo) and arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(caminho_completo)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagens.append(img)
            nomes_arquivos.append(arquivo)

caracteristicas = []
for img in imagens:
    intensidade_media = np.mean(img, axis=(0, 1))
    caracteristicas.append(intensidade_media)

from sklearn.cluster import KMeans

k = 3 

modelo = KMeans(n_clusters=k)
grupos = modelo.fit_predict(caracteristicas)

resultados = [[] for _ in range(k)]
for i, grupo in enumerate(grupos):
    resultados[grupo].append(imagens[i])

grupos_de_imagens = {i: [] for i in range(k)}
for i, grupo in enumerate(grupos):
    nome_imagem = nomes_arquivos[i]  
    grupos_de_imagens[grupo].append(nome_imagem) 

for grupo, nomes in grupos_de_imagens.items():
    print(f"Grupo {grupo}: Imagens {nomes}")