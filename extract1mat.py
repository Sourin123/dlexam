import cv2

def extract_binary_pixel_values(image_path):
    # Read the image in grayscale mode
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray_image is None:
        print("Error: Unable to load image.")
        return

    # Get the dimensions of the image
    height, width = gray_image.shape
    print(f"Image dimensions: {width}x{height}")

    # Iterate through each pixel and get its binary value
    binary_pixel_values = []
    for y in range(height):
        row = []
        for x in range(width):
            # Get the pixel value and convert to 8-bit binary
            binary_value = format(gray_image[y, x], '08b')
            row.append(binary_value)
        binary_pixel_values.append(row)

    return binary_pixel_values

# Specify the path to the graymap image
image_path = "image.png"

# Extract binary pixel values
binary_values = extract_binary_pixel_values(image_path)

if binary_values:
    for row in binary_values:
        print(" ".join(row))
