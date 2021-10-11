#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sep 30 2021
@author: zac.liu

Utility script for edge case detection
1. transform image data into embedding cluster using img2vdc and tSNE
2. apply isolation forest to embeddings to filter our outliers

"""

from img2vec_keras import Img2Vec
import glob
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import cv2

img2vec = Img2Vec()

def edge_case(image_path, image_classes, outlier_path):
	
	# Set up image path
	image_paths = []
	#for image_class in image_classes:
	image_paths.extend(glob.glob(image_path + image_classes + '/*.png'))
	print('Found ', len(image_paths), ' images in class ', image_classes)

	# Use img2vec to convert image to vectors 
	image_vectors = {}
	for image_path in image_paths:
		vector = img2vec.get_vec(image_path)
		image_vectors[image_path] = vector

	# Use PCA to reduce dimensionality
	X = np.stack(list(image_vectors.values()))
	pca_50 = PCA(n_components=50)
	pca_result_50 = pca_50.fit_transform(X)
	print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
	print(np.shape(pca_result_50))

	# tSNE transformation
	tsne = TSNE(n_components=2, verbose=1, n_iter=3000)
	tsne_result = tsne.fit_transform(pca_result_50)
	tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

	# Apply Isolation Forest for outlier detection
	clf = IsolationForest(random_state=123)
	preds = clf.fit_predict(tsne_result_scaled)

	# copy outlier image to another directory
	outlier_paths = []
	count = 0

	for image_path in image_paths:
		if preds[count] == -1:
			outlier_paths_c = shutil.copy(image_path, outlier_path + image_classes)
			outlier_paths.append(outlier_paths_c)  
		count = count + 1

	print('Found ', len(outlier_paths), ' images as outliers')
