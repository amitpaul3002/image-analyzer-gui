import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
def detect_copy_move(image_path , min_cluster_size, sensitivity=1):
    img = cv2.imread(image_path)
    image = remove_noise(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < sensitivity * n.distance:
            good_matches.append(m)
    if len(good_matches) < min_cluster_size:
        return "Original"
    keypoints1 = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
    clustering = KMeans(n_clusters=2, n_init=10).fit(keypoints1)
    unique_clusters = set(clustering.labels_)
    if len(unique_clusters) > 1:
        return "Forged"
    else:
        return "Original"

def remove_noise(image):
    if image is not None:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    else:
        return None

