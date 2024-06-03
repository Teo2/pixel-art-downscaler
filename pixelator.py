#import matplotlib
#matplotlib.use('TkAgg')  # Set the backend to TkAgg for interactive plotting
import math

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.fft import fft, fftfreq
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import random
import mplcursors
from mediancut import median_cut
import time
import argparse


class ColorPalette:
    def __init__(self):
        self.colors = None

    def k_means_palette_from_image(self, image, num_colors):
        # Reshape image array to 2D array
        pixels = image.reshape((-1, 3))

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)

        # Replace pixel values with cluster centroids
        return kmeans.cluster_centers_

    def median_cut_palette_from_image(self, image, num_colors):
        mcut = median_cut(image, num_colors)
        array = np.array(mcut)
        return array


    def generate_palette_from_image(self, method, image, num_colors, color_space = "RGB"):
        color_space_conversion = {
            "RGB": (False, False),  # Identity conversion
            "LAB": (cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB),
            "HSV": (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
            # Add more color spaces as needed
        }

        if color_space not in color_space_conversion:
            raise ValueError("Unsupported color space")

        convert_to, convert_back = color_space_conversion[color_space]

        # Convert image to the specified color space
        converted_image = np.array(image)
        if (convert_to):
            converted_image = cv2.cvtColor(converted_image, convert_to)


        # Perform the palettization in the specified color space
        if method == 'k_means':
            color_list = self.k_means_palette_from_image(converted_image, num_colors)
        elif method == 'median_cut':
            color_list = self.median_cut_palette_from_image(converted_image, num_colors)
        else:
            raise ValueError("Unknown palette generation method")

        # Convert the color list back to RGB
        if (convert_back):
            color_list = np.array([cv2.cvtColor(np.uint8([[color]]), convert_back)[0][0] for color in color_list])

        self.colors = color_list

    def set_palette(self, colors):
        self.colors = colors

    def apply_palette_to_image(self, image):
        # Convert image to numpy array
        image_np = np.array(image)
        # Reshape image array to 2D array
        pixels = image_np.reshape((-1, 3))

        # Build a KD tree from the palette colors
        kdtree = KDTree(self.colors.astype('uint8'))

        # Query the KD tree to find the nearest color in the palette for each pixel
        _, closest_indices = kdtree.query(pixels)

        # Get the colors from the palette corresponding to the closest indices
        new_pixels = self.colors.astype('uint8')[closest_indices]

        # Reshape pixels back to original image shape
        new_image_np = new_pixels.reshape(image_np.shape).astype('uint8')

        # Convert numpy array back to PIL Image
        new_image = Image.fromarray(new_image_np)

        return new_image


def canny(input):
    image = pillow_to_cv2(input)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)  # Adjust the thresholds as needed


    # Display the original and edge-detected images
    cv2.imshow('Original Image', gray_image)
    cv2.imshow('Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edges

def sobel(input):
    image = pillow_to_cv2(input)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize the gradient magnitude to the range [0, 255]
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the edges to uint8 type
    edges = np.uint8(edges)

    # Convert to grayscale (keep only one channel)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    return edges

def pillow_to_cv2(pillow_image):
    # Convert Pillow image to NumPy array
    cv2_image = np.array(pillow_image)
    # Convert RGB to BGR (OpenCV uses BGR order)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    return cv2_image


def cv2_to_pillow(cv2_image):
    # Convert OpenCV image to RGB format
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert to Pillow image
    pillow_image = Image.fromarray(cv2_image_rgb)

    return pillow_image

def sobel_color(input, ksize=1):
    image = pillow_to_cv2(input)

    # Convert the image to the LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into individual channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply Sobel edge detection to each color channel
    sobel_x_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y_l = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    edges_l = np.sqrt(sobel_x_l ** 2 + sobel_y_l ** 2)

    sobel_x_a = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y_a = cv2.Sobel(a_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    edges_a = np.sqrt(sobel_x_a ** 2 + sobel_y_a ** 2)

    sobel_x_b = cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y_b = cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    edges_b = np.sqrt(sobel_x_b ** 2 + sobel_y_b ** 2)

    # Combine the edges from each channel
    edges = cv2.normalize(edges_l + edges_a + edges_b, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the edges to uint8 type
    edges = np.uint8(edges)

    return edges


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def scale_factor(edges):
    edges_mean_subtracted = edges - np.mean(edges, axis=1, keepdims=True)

    # Perform Fourier Transform on each row
    fft_rows = np.fft.fft(edges_mean_subtracted, axis=1)

    # Sum the results
    summed_fft = np.sum(np.abs(fft_rows), axis=0)

    # Plot the results
    frequencies = np.fft.fftfreq(len(summed_fft))


    #damping_factor = np.abs(frequencies)
    damping_factor = 1 / (142000 * 0.076 / (np.abs(frequencies) + 0.011) + 78100)
    summed_fft_damped = summed_fft * damping_factor

    max_index = np.argmax(summed_fft_damped)

    # Get the frequency corresponding to the maximum value
    max_frequency = frequencies[max_index]
    return clamp(abs(max_frequency), 1/10000, 1)
    #pixel_size = 1 / max_frequency
    #return pixel_size

def gaussian_kernel(size, sigma=1):
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def smart_nn_downscale(input, edges, new_width, new_height, edge_avoidance_factor = 1, chosen_points_path = None):
    image = np.array(input)
    kernel_size = 7 #should probably always be odd, or I should adjust gaussian_kernel function
    padding = kernel_size // 2
    padded_interiors = np.pad(- (edges.astype(np.float32) / 255) ** 2, padding, mode='constant', constant_values=0 if edge_avoidance_factor == 0 else -np.inf)
    kernel = gaussian_kernel(kernel_size, 2)

    old_height, old_width, num_channels = image.shape

    result = np.zeros((new_height, new_width, num_channels), dtype=image.dtype)

    # Compute downscale factor
    height_ratio = old_height / new_height
    width_ratio = old_width / new_width

    chosen_point = np.copy(image)

    for y in range(new_height):
        for x in range(new_width):
            old_y = int((y + 0.5) * height_ratio)
            old_x = int((x + 0.5) * width_ratio)

            submatrix = padded_interiors[old_y:old_y + kernel_size, old_x:old_x + kernel_size]

            weighted_submatrix = submatrix * edge_avoidance_factor + kernel
            offset_y, offset_x = np.unravel_index(np.argmax(weighted_submatrix), weighted_submatrix.shape)
            offset_x -= padding
            offset_y -= padding

            chosen_point[old_y, old_x] = [0, 0, 255]
            chosen_point[old_y + offset_y, old_x + offset_x] = [255,0,0]
            result[y, x] = image[old_y + offset_y, old_x + offset_x]

    if (chosen_points_path != None):
        Image.fromarray(chosen_point).save(chosen_points_path)


    return Image.fromarray(result)

def nn_downscale(input, new_width, new_height):
    new_size = (width, height)
    return image.resize(new_size, Image.NEAREST)


def process_image(input_path, output_path, scale_factor_value=None, num_colors=None):
    image = Image.open(input_path)

    # Apply sobel edge detection
    edges = sobel_color(image, ksize=1)

    # Use specified scale factor or calculate based on edges
    if scale_factor_value:
        new_width = int(image.width / scale_factor_value)
        new_height = int(image.height / scale_factor_value)
    else:
        scale_factor_value = scale_factor(edges)
        new_width = int(image.width * scale_factor_value)
        new_height = int(image.height * scale_factor_value)

    # Downscale the image using smart nearest neighbor downscaling
    resized_image = smart_nn_downscale(image, edges, new_width, new_height, edge_avoidance_factor=1)

    # Save the resized image
    resized_image.save(output_path, format="PNG")

    # Generate and apply a color palette if num_colors is provided
    if num_colors:
        palette = ColorPalette()
        palette.generate_palette_from_image('k_means', resized_image, num_colors)
        palletized_image = palette.apply_palette_to_image(resized_image)

        # Save the palette-applied image
        palletized_image.save(output_path, format="PNG")


def main():
    parser = argparse.ArgumentParser(description='Downscale pixel art images and restrict the color palette.')
    parser.add_argument('-i', dest='input_path', type=str, help='Path to the input image.')
    parser.add_argument('-o', dest='output_path', type=str,
                        help='Path to the output image. If not provided, the output image will be saved as "<input_image_path>_pixelated.png".')
    parser.add_argument('-s', dest='scale_factor', type=float, help='Manually set the scale factor for downsampling.')
    parser.add_argument('-c', dest='num_colors', type=int, help='Number of colors in the final image.')
    args = parser.parse_args()

    input_path = args.input_path

    # Determine the output image path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = input_path.rsplit('.', 1)[0] + "_pixelated.png"

    process_image(input_path, output_path, args.scale_factor, args.num_colors)


if __name__ == "__main__":
    main()


