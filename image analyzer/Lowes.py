import cv2
import numpy as np

def divide_into_blocks(image, block_size):
    height, width = image.shape[:2]
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size
    blocks = []
    block_coordinates = []
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            start_y = i * block_size
            end_y = (i + 1) * block_size
            start_x = j * block_size
            end_x = (j + 1) * block_size
            block = image[start_y:end_y, start_x:end_x]
            blocks.append(block)
            block_coordinates.append((start_x, start_y, end_x, end_y))
    return blocks, block_coordinates

def calculate_variance(block):
    block = np.float32(block)
    variance = cv2.meanStdDev(block)[1][0] ** 2
    return variance

def calculate_lowes_ratio(block, blocks,):
    min_variance = float('inf')
    for neighbor_block in blocks:
        neighbor_variance = calculate_variance(neighbor_block)
        if neighbor_variance < min_variance:
            min_variance = neighbor_variance
    block_variance = calculate_variance(block)
    lowes_ratio = block_variance / min_variance
    return lowes_ratio


def lowes_ratio(image_path,threshold, block_size=32, search_window_size=64):
    image=cv2.imread(image_path)
    blocks, block_coordinates = divide_into_blocks(image, block_size)
    for i, block in enumerate(blocks):
        start_x = max(0, block_coordinates[i][0] - search_window_size)
        start_y = max(0, block_coordinates[i][1] - search_window_size)
        end_x = min(image.shape[1], block_coordinates[i][2] + search_window_size)
        end_y = min(image.shape[0], block_coordinates[i][3] + search_window_size)
        neighbor_blocks = [blocks[j] for j in range(len(blocks)) if
                           start_x <= block_coordinates[j][0] <= end_x and start_y <= block_coordinates[j][1] <= end_y]
        lowes_ratio = calculate_lowes_ratio(block, neighbor_blocks)
        if lowes_ratio < threshold:
            return "Forged"
        else:
            return "Original"

