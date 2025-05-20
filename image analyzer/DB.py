import cv2
import numpy as np
from sklearn.cluster import DBSCAN
def siftDetector(image_path):
    image = cv2.imread(image_path)
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptors = sift.detectAndCompute(gray, None)
    return key_points, descriptors
def ShowForgery(image_path, key_points, descriptors, eps, min_sample=2):
    image = cv2.imread(image_path)
    clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(descriptors)
    size = np.unique(clusters.labels_).shape[0] - 1
    forgery = image.copy()

    if size == 0 and np.unique(clusters.labels_)[0] == -1:
        return "Original", image

    if size == 0:
        size = 1

    cluster_list = [[] for _ in range(size)]
    for idx in range(len(key_points)):
        if clusters.labels_[idx] != -1:
            cluster_list[clusters.labels_[idx]].append((int(key_points[idx].pt[0]), int(key_points[idx].pt[1])))

    for points in cluster_list:
        if len(points) > 1:
            for idx1 in range(1, len(points)):
                cv2.line(forgery, points[0], points[idx1], (255, 0, 0), 5)
                pt1, pt2 = points[0], points[idx1]
                line_length = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
                if line_length > 1:
                    return "Forged", forgery

    return "Original", image
def locateForgery(key_points, descriptors, eps=21, min_sample=2):
    clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(descriptors)
    size = np.unique(clusters.labels_).shape[0] - 1
   
    if size == 0 and np.unique(clusters.labels_)[0] == -1:
        return "Original"
    if size == 0:
        size = 1
    cluster_list = [[] for _ in range(size)]
    for idx in range(len(key_points)):
        if clusters.labels_[idx] != -1:
            cluster_list[clusters.labels_[idx]].append((int(key_points[idx].pt[0]), int(key_points[idx].pt[1])))
    for points in cluster_list:
        if len(points) > 1:
            for idx1 in range(1, len(points)):
                pt1, pt2 = points[0], points[idx1]
                line_length = np.sqrt((pt2[0] - pt1[0])*2 + (pt2[1] - pt1[1])*2)
                if line_length > 1:
                    return "Forged"
    return "Original"

