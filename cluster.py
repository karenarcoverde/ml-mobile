from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import os

# Caminho para a pasta com as imagens
folder_path = 'dataset'

# Carregar imagens e armazenar nomes
images = []
image_names = []
for filename in os.listdir(folder_path):
    if filename.endswith('.jpeg'):
        img = Image.open(os.path.join(folder_path, filename))
        img = img.convert('L')  # Converter para escala de cinza
        img = img.resize((100, 100))  # Redimensionar
        images.append(np.array(img).flatten())
        image_names.append(filename)  # Armazenar o nome do arquivo

# Normalizar os vetores de imagem
scaler = StandardScaler()
images_scaled = scaler.fit_transform(images)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=2, algorithm='kd_tree')
clusters = dbscan.fit_predict(images_scaled)

# Associar nomes das imagens aos clusters
image_clusters = {name: cluster for name, cluster in zip(image_names, clusters)}
print(clusters)