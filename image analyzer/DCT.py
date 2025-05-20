import cv2
import numpy as np
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where n * nrows * ncols = arr.size
    """
    h, w = arr.shape
    return arr[:h // nrows * nrows, :w // ncols * ncols].reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)
def detect_copy_move(image_path, threshold, block_size=8):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    blocks = blockshaped(image, block_size, block_size)
    dct_blocks = np.zeros_like(blocks, dtype=np.float32)
    for i in range(len(blocks)):
        dct_blocks[i] = cv2.dct(np.float32(blocks[i]))
    flattened_dct = dct_blocks.reshape(dct_blocks.shape[0], -1)
    hashes = [hash(tuple(block.flatten())) for block in flattened_dct]
    count =0
    for i in range(len(hashes)):
        for j in range(i+1, len(hashes)):
            if abs(hashes[i] - hashes[j]) < 10: 
                count += 1
    if count>threshold:
        return "Forged"
    else:
        return "Original"