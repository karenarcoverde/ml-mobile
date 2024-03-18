import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import Birch
from sklearn.metrics import f1_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.decomposition import PCA


def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    return features.flatten()

def predict_clusters(image_paths, kmeans_model):
    feature_list = [extract_features(img_path) for img_path in image_paths]
    feature_array = np.array(feature_list)
    predictions = kmeans_model.predict(feature_array)
    return predictions


data = pd.read_excel('./predict_correct1.xlsx')
accuracy = accuracy_score(data['correct'], data['predict'])
print(accuracy)

y_pred = data['predict']
y_true = data['correct']

f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
print(f1_micro)

image.LOAD_TRUNCATED_IMAGES = True 
model = VGG16(weights='imagenet', include_top=False)

imdir = './dataset/'
targetdir = "./outdir/"
number_clusters = 3

filelist = glob.glob(os.path.join(imdir, '*.jpeg'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

featurelist = np.array(featurelist).astype('float64')
pca = PCA(n_components=2)  # You can vary n_components to test different levels of dimensionality reduction
pca_featurelist = pca.fit_transform(featurelist)
# # Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(pca_featurelist))

# Copy images renamed by cluster 
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    #print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpeg")

featurelist_array = np.array(featurelist, dtype=np.float32)

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init = "k-means++", n_init = 10, max_iter = 100, algorithm = "elkan" ).fit(pca_featurelist)
    score = silhouette_score(pca_featurelist, kmeans.labels_)
    silhouette_scores.append(score)

print("KMEANS - " +str(silhouette_scores))

silhouette_scores = []

for n_clusters in range(2, 11):
    birch = Birch(n_clusters=n_clusters, threshold=0.7, branching_factor = 20).fit(pca_featurelist)
    labels = birch.labels_
    silhouette_avg = silhouette_score(pca_featurelist, labels)
    silhouette_scores.append(silhouette_avg)

print("birch - " +str(silhouette_scores))

silhouette_scores = []
for n_clusters in range(2, 11):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglomerative.fit(pca_featurelist)

    labels = agglomerative.labels_

    silhouette_avg = silhouette_score(pca_featurelist, labels)
    silhouette_scores.append(silhouette_avg)

print("AGGLOMERATIVE - " +str(silhouette_scores))


silhouette_scores = []
for n_clusters in range(2, 11):
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='nearest_neighbors')
    labels = spectral.fit_predict(pca_featurelist)

    silhouette_avg = silhouette_score(pca_featurelist, labels)
    silhouette_scores.append(silhouette_avg)

print("SPECTRAL - " +str(silhouette_scores))
