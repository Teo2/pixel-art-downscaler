import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def quantize_image(image, num_colors):
    # Convert image to numpy array
    image_np = np.array(image)
    # Reshape image array to 2D array
    pixels = image_np.reshape((-1, 3))

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Replace pixel values with cluster centroids
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape pixels back to original image shape
    quantized_image_np = new_pixels.reshape(image_np.shape).astype('uint8')

    # Convert numpy array back to PIL Image
    quantized_image = Image.fromarray(quantized_image_np)

    return quantized_image


if __name__ == "__main__":
    # Example usage
    input_image_path = "..\\pixel1.jpeg"
    output_path = "..\\final_image.png"
    scale_factor = 0.5  # Scale down by 50%
    num_colors = 64  # Increase the number of colors for better representation

    # Open image
    image = Image.open(input_image_path)
    # Resize image
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    new_size = (new_width, new_height)
    resized_image = image.resize(new_size, resample=Image.NEAREST)
    # Quantize image with k-means clustering
    quantized_image = quantize_image(resized_image, num_colors)
    # Save the final image as PNG
    quantized_image.save(output_path, format="PNG")