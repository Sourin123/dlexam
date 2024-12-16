import numpy as np
import cv2
import matplotlib.pyplot as plt

def pool_image(image, pool_size, pool_type='max'):
    if not isinstance(image, np.ndarray):
        print("Error: Input image must be a NumPy array.")
        return None
    if image.ndim != 2:
        print("Error: Input image must be 2-dimensional (grayscale).")
        return None
    if pool_size <= 0 or not isinstance(pool_size, int):
        print("Error: Pool size must be a positive integer.")
        return None
    if pool_type not in ['max', 'min', 'average']:
        print("Error: Invalid pooling type. Choose 'max', 'min', or 'average'.")
        return None

    rows, cols = image.shape
    pooled_rows = rows // pool_size
    pooled_cols = cols // pool_size

    pooled_image = np.zeros((pooled_rows, pooled_cols), dtype=image.dtype)

    for i in range(pooled_rows):
        for j in range(pooled_cols):
            window = image[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]

            if pool_type == 'max':
                pooled_image[i, j] = np.max(window)
            elif pool_type == 'min':
                pooled_image[i, j] = np.min(window)
            elif pool_type == 'average':
                pooled_image[i, j] = np.mean(window)

    return pooled_image

# Load the grayscale image
image_path = "./LeNet/luffy-pfp-4403.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image could not be loaded. Check the file path.")
else:
    pool_size = 2

    # Perform pooling
    max_pooled = pool_image(image, pool_size, 'max')
    min_pooled = pool_image(image, pool_size, 'min')
    avg_pooled = pool_image(image, pool_size, 'average')

    # Display the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')

    if max_pooled is not None:
        plt.subplot(1, 4, 2)
        plt.imshow(max_pooled, cmap='gray')
        plt.title('Max Pooling')

    if min_pooled is not None:
        plt.subplot(1, 4, 3)
        plt.imshow(min_pooled, cmap='gray')
        plt.title('Min Pooling')

    if avg_pooled is not None:
        plt.subplot(1, 4, 4)
        plt.imshow(avg_pooled, cmap='gray')
        plt.title('Average Pooling')

    plt.show()
