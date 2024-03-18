import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, glob, os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.cluster import KMeans
import os, glob
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score
from sklearn.cluster import Birch
import shutil

image.LOAD_TRUNCATED_IMAGES = True 
model = VGG16(weights='imagenet', include_top=False)

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
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(featurelist)

kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(reduced_features))
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

#### treino

cluster_labels = kmeans.labels_
custom_cmap = ListedColormap(['red', 'orange', 'blue'])
plt.figure(figsize=(12, 12))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap=custom_cmap)

plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel('PCA Feature 1', fontsize=14)
plt.ylabel('PCA Feature 2', fontsize=14)

legend_labels = {0: 'Ruim', 1: 'Médio', 2: 'Bom'}
handles, _ = scatter.legend_elements()
legend = plt.legend(handles, [legend_labels[i] for i in range(len(legend_labels))], title="Clusters")

plt.setp(legend.get_title(), fontsize=16)

plt.show()

### teste
imdir = './new_dataset'

filelist = glob.glob(os.path.join(imdir, '*.jpeg'))
filelist.sort()

new_feature_list = [extract_features(img_path) for img_path in filelist]
new_feature_array = np.array(new_feature_list, dtype=np.float32)



pca = PCA(n_components=2)
reduced_new_features = pca.fit_transform(new_feature_array)
reduced_new_features = reduced_new_features.astype(np.float64)
cluster_predictions = kmeans.predict(reduced_new_features)
for img_path, cluster in zip(filelist, cluster_predictions):
    print(f'{os.path.basename(img_path)} belongs to cluster {cluster}')

custom_cmap = ListedColormap(['red', 'orange', 'blue'])



plt.figure(figsize=(12, 12))
scatter = plt.scatter(reduced_new_features[:, 0], reduced_new_features[:, 1], c=cluster_predictions, cmap=custom_cmap)

plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel('PCA Feature 1', fontsize=14)
plt.ylabel('PCA Feature 2', fontsize=14)

legend_labels = {0: 'Ruim', 1: 'Médio', 2: 'Bom'}
handles, _ = scatter.legend_elements()
legend = plt.legend(handles, [legend_labels[i] for i in range(len(legend_labels))], title="Clusters")

plt.setp(legend.get_title(), fontsize=16)

plt.show()

#### teste silhueta
featurelist_array = np.array(featurelist, dtype=np.float32)
new_feature_list = [extract_features(img_path) for img_path in filelist]
new_featurelist_array = np.array(new_feature_list, dtype=np.float32)
pca = PCA(n_components=2)  # You can vary n_components to test different levels of dimensionality reduction
pca_featurelist = pca.fit_transform(new_featurelist_array)

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init = "k-means++", n_init = 10, max_iter = 100, algorithm = "elkan" ).fit(pca_featurelist)
    predictions = kmeans.predict(pca_featurelist)
    score = silhouette_score(pca_featurelist, predictions)
    silhouette_scores.append(score)

print("KMEANS - " +str(silhouette_scores))

featurelist_array = np.array(featurelist, dtype=np.float32)

silhouette_scores = []

for n_clusters in range(2, 11):
    birch = Birch(n_clusters=n_clusters, threshold=0.7, branching_factor = 20).fit(pca_featurelist)
    predictions = birch.predict(pca_featurelist)

    silhouette_avg = silhouette_score(pca_featurelist, predictions)
    silhouette_scores.append(silhouette_avg)

print("birch - " +str(silhouette_scores))
