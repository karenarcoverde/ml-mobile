import pandas as pd
from sklearn.metrics import accuracy_score
data = pd.read_excel('./predict_correct.xlsx')
accuracy = accuracy_score(data['correct'], data['predict'])
print(accuracy)

correlation = data['predict'].corr(data['correct'])
print(correlation)

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True 
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = './dataset/'
targetdir = "./outdir/"
number_clusters = 3

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpeg'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

# Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

# Copy images renamed by cluster 
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpeg")

from sklearn.metrics import silhouette_score


from sklearn.cluster import Birch
featurelist_array = np.array(featurelist)

silhouette_scores = []
for n_clusters in range(2, 11):  # De 2 a 10 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(featurelist_array)
    score = silhouette_score(featurelist_array, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"Índice Médio de Silhueta para {n_clusters} clusters: {score}")

# O resultado será uma lista de valores do Índice de Silhueta para cada número de clusters
print(silhouette_scores)

featurelist_array = np.array(featurelist)

# Armazenar os valores do Índice Médio de Silhueta para diferentes números de clusters
silhouette_scores = []

# Calculando o Índice de Silhueta para número de clusters de 2 a 10
for n_clusters in range(2, 11):
    birch = Birch(n_clusters=n_clusters, threshold=0.5).fit(featurelist_array)
    labels = birch.labels_
    silhouette_avg = silhouette_score(featurelist_array, labels)
    silhouette_scores.append(silhouette_avg)

print(silhouette_scores)


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

# Supondo que 'featurelist_array' contém as características extraídas das imagens
featurelist_array = np.array(featurelist)

silhouette_scores = []
for n_clusters in range(2, 11):
    # Aplicar o Clustering Hierárquico Aglomerativo
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglomerative.fit(featurelist_array)

    # Rótulos dos clusters
    labels = agglomerative.labels_

    # Calcular o Índice Médio de Silhueta
    silhouette_avg = silhouette_score(featurelist_array, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Índice Médio de Silhueta para {n_clusters} clusters: {silhouette_avg}")

# O resultado será uma lista de valores do Índice de Silhueta para cada número de clusters
print(silhouette_scores)


from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np

# Supondo que 'featurelist_array' contém as características extraídas das imagens
featurelist_array = np.array(featurelist)

silhouette_scores = []
for n_clusters in range(2, 11):
    # Aplicar Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(featurelist_array)
    labels = gmm.predict(featurelist_array)

    # Calcular o Índice Médio de Silhueta
    silhouette_avg = silhouette_score(featurelist_array, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Índice Médio de Silhueta para {n_clusters} clusters: {silhouette_avg}")

# O resultado será uma lista de valores do Índice de Silhueta para cada número de clusters
print(silhouette_scores)

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np

# Supondo que 'featurelist_array' contém as características extraídas das imagens
featurelist_array = np.array(featurelist)

silhouette_scores = []
for n_clusters in range(2, 11):
    # Aplicar o Clustering Espectral
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='nearest_neighbors')
    labels = spectral.fit_predict(featurelist_array)

    # Calcular o Índice Médio de Silhueta
    silhouette_avg = silhouette_score(featurelist_array, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Índice Médio de Silhueta para {n_clusters} clusters: {silhouette_avg}")

# O resultado será uma lista de valores do Índice de Silhueta para cada número de clusters
print(silhouette_scores)

from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
import numpy as np

# Supondo que 'featurelist_array' contém as características extraídas das imagens
featurelist_array = np.array(featurelist)

# Aplicar o MeanShift
meanshift = MeanShift(bandwidth=None)  # Bandwidth pode ser ajustado se necessário
meanshift.fit(featurelist_array)

# Rótulos dos clusters
labels = meanshift.labels_

# Calcular o Índice Médio de Silhueta
# Deve-se ter pelo menos 2 clusters (excluindo ruído) para calcular o índice
if len(set(labels)) > 1:
    silhouette_avg = silhouette_score(featurelist_array, labels)
    print("Índice Médio de Silhueta:", silhouette_avg)
else:
    print("Não foi possível calcular o Índice de Silhueta com um único cluster.")
