import cv2
import numpy as np

def detect_orb_flann(image_path,min_matches, percentile=95,  block_size=64):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    orb = cv2.ORB_create(nfeatures=3000)  
    all_keypoints = []
    all_descriptors = []
    
    for y in range(0, img.shape[0], block_size):
        for x in range(0, img.shape[1], block_size):
            block = img[y:y + block_size, x:x + block_size]
            keypoints, descriptors = orb.detectAndCompute(block, None)
            if descriptors is not None:
                for kp in keypoints:
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                all_keypoints.extend(keypoints)
                all_descriptors.append(descriptors)
    
    if len(all_descriptors) == 0:
        return "Original"
    descriptors = np.vstack(all_descriptors)
    
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(descriptors, descriptors)
    
    if len(matches) == 0:
        return "Original"
    
    matches = sorted(matches, key=lambda x: x.distance)
    filtered_matches = [m for m in matches] 
    matched_keypoints = []
    
    for match in filtered_matches:
        matched_keypoints.append(all_keypoints[match.queryIdx].pt)
        matched_keypoints.append(all_keypoints[match.trainIdx].pt)
    
    matched_keypoints = np.asarray(matched_keypoints)
    
    if len(matched_keypoints) == 0:
        return "Original"
    
    distances = np.sqrt(np.sum(np.square(matched_keypoints[:, None] - matched_keypoints), axis=-1))
    threshold = np.percentile(distances, percentile)
    close_keypoints = distances < threshold
    close_counts = np.sum(close_keypoints, axis=-1)
    
    if np.max(close_counts) > min_matches:
        return "Forged"
    
    return "Original"
