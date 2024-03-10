from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from skimage import io
import cv2

# Variables
imdir = './dataset/'
targetdir = "./dataset_clusters_dominant/"
number_clusters = 3

# Build reference image
image_left = cv2.imread(os.path.join(imdir, 'heatmap - 2024-02-03T172930.818.jpeg'))

image_right = cv2.imread(os.path.join(imdir, 'heatmap - 2024-02-03T172310.951.jpeg'))

ref_image = np.zeros((501, 1496, 3), dtype=np.uint8)
ref_image[:,:789,:] = image_left[:,:789,:]
ref_image[:,789:,:] = image_right[:,789:,:]

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpeg'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    image = cv2.imread(imagepath) - ref_image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    cluster = KMeans(n_clusters=5, n_init=10).fit(reshape)
    centroids = cluster.cluster_centers_

    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end

    featurelist.append(rect.flatten())

# Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0, n_init=10).fit(np.array(featurelist))

# Copy images renamed by cluster 
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpeg")