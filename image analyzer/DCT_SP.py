import cv2
import numpy as np
import matplotlib.pyplot as plt

def dct(image):
    return cv2.dct(np.float32(image))
def idct(dct_image):
    return cv2.idct(dct_image)
def hpf(dct_image, threshold=50):
    dct_filtered = dct_image.copy()
    rows, cols = dct_filtered.shape
    for i in range(rows):
        for j in range(cols):
            if abs(dct_filtered[i, j]) < threshold:
                dct_filtered[i, j] = 0
    return dct_filtered

def forgery(image_path):
    # Step 1: Take an image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "Image not found or invalid image format."
    
    # Step 2: Apply DCT
    dct_image = dct(image)
    
    # Step 3: Pass High-Pass Filter
    hpf_dct_image = hpf(dct_image)
    
    # Step 4: Apply Inverse DCT
    filtered_image = idct(hpf_dct_image)
    
    # Step 5: Determine Forged Area
    diff_image = np.abs(image - filtered_image)
    _, forged_area = cv2.threshold(diff_image, 50, 255, cv2.THRESH_BINARY)
    
    # Visualize the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Filtered Image")
    plt.imshow(filtered_image, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Detected Forged Area")
    plt.imshow(forged_area, cmap='gray')
    plt.show()
    
    # Step 6: Return Forged or Original
    if np.sum(forged_area) > 0:
        return "Forged"
    else:
        return "Original"

# Example usage
image_path = 'C:\\Users\\ALONE\\.vscode\\.vscode\\dataset\\MICC-F220\\CRW_4809_scale.jpg'
result = forgery(image_path)
print(f"The image is: {result}")
