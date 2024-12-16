import cv2
import numpy as np

def extract_bit_planes(image_path):
    # Read the image in grayscale mode
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray_image is None:
        print("Error: Unable to load image.")
        return

    # Get the dimensions of the image
    height, width = gray_image.shape
    print(f"Image dimensions: {width}x{height}")

    # Create 8 matrices for each bit plane
    bit_planes = []
    for bit in range(8):
        # Extract the bit plane by bitwise AND and shifting
        bit_plane = (gray_image >> bit) & 1
        bit_planes.append(bit_plane)
    
    return bit_planes

# Specify the path to the graymap image
image_path = "image.png"

# Extract bit planes
bit_planes = extract_bit_planes(image_path)

if bit_planes:
    for i, plane in enumerate(bit_planes):
        print(f"Bit plane {i}:")
        print(plane)
