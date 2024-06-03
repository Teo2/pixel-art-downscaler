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
import os
from scipy.interpolate import interp1d

class ColorNode:
    def __init__(self, colors):
        self.colors = colors
        self.children = []

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

def palletize(input_image_path, output_path):
    pixel_block_size = 8;
    scale_factor = 1 / pixel_block_size;
    num_colors = 16

    # Open image
    image = Image.open(input_image_path)
    # Resize image
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    new_size = (new_width, new_height)
    resized_image = image.resize(new_size, resample=Image.NEAREST)

    # Create ColorPalette instance
    palette = ColorPalette()
    # Generate palette using K-Means from the resized image
    palette.generate_palette_from_image('k_means', resized_image, num_colors)

    # Apply palette to resized image
    new_image = palette.apply_palette_to_image(resized_image)

    # Save the final image
    new_image.save(output_path, format="PNG")

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

def high_pass_filter(image, sigma=3):
    # Apply Gaussian filter to smooth the image
    smoothed = ndimage.gaussian_filter(image, sigma)

    # Subtract the smoothed image from the original to retain high-frequency components
    highpass = image - smoothed

    return highpass

def fourier_transform_and_plot(edges):
    edges_mean_subtracted = edges - np.mean(edges, axis=1, keepdims=True)

    # Apply high-pass filter to remove low-frequency components
    #edges_highpass = high_pass_filter(edges, 3)

    # Perform Fourier Transform on each row
    fft_rows = np.fft.fft(edges_mean_subtracted, axis=1)

    # Sum the results
    summed_fft = np.sum(np.abs(fft_rows), axis=0)

    # Plot the results
    frequencies = np.fft.fftfreq(len(summed_fft))

    damping_factor = 1 / (142000 * 0.076 / (np.abs(frequencies) + 0.011) +78100)
    summed_fft_damped = summed_fft * damping_factor

    plt.plot(frequencies, summed_fft_damped)
    plt.xlabel('Frequency')
    plt.ylabel('Sum of Fourier Transform')
    plt.title('Fourier Transform Sum')
    plt.show()

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def scale_factor(edges):
    edges_mean_subtracted = edges - np.mean(edges, axis=1, keepdims=True)

    # Apply high-pass filter to remove low-frequency components
    edges_highpass = high_pass_filter(edges, 3)

    # Perform Fourier Transform on each row
    fft_rows = np.fft.fft(edges_mean_subtracted, axis=1)

    # Sum the results
    summed_fft = np.sum(np.abs(fft_rows), axis=0)

    # Plot the results
    frequencies = np.fft.fftfreq(len(summed_fft))


    damping_factor = 1 / (142000 * 0.076 / (np.abs(frequencies) + 0.011) +78100)
    summed_fft_damped = summed_fft * damping_factor

    max_index = np.argmax(summed_fft_damped)

    # Get the frequency corresponding to the maximum value
    max_frequency = frequencies[max_index]
    return clamp(abs(max_frequency), 1/10000, 1)
    #pixel_size = 1 / max_frequency
    #return pixel_size

def plot_intensity(intensity):
    # Plot the intensity of the row
    plt.subplot(2, 1, 1)
    plt.plot(intensity)
    plt.xlabel('Pixel Position')
    plt.ylabel('Intensity')
    plt.title(f'Intensity')

    # Perform Fourier Transform on the row
    row_fft = np.fft.fft(intensity)

    # Calculate the corresponding frequency axis
    num_samples = len(row_fft)
    pixel_spacing = 0.1
    sample_rate = 1 / pixel_spacing
    frequencies = np.fft.fftfreq(num_samples, d=1/sample_rate)

    # Plot the Fourier Transform of the row
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, np.abs(row_fft))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title(f'FFT')

    plt.tight_layout()
    plt.show()

def find_seam(edges, end = None):
    change_col_penalty = 1500;

    goodness = np.zeros(edges.shape, dtype=float)
    parent = np.zeros(edges.shape, dtype=int)
    for row in range(edges.shape[0]):
        for col in range(edges.shape[1]):
            left = (row - 1, col - 1)
            mid = (row - 1, col)
            right = (row - 1, col + 1)
            here = (row, col)

            left_side = col == 0
            right_side = col == edges.shape[1] - 1

            left_choice = -math.inf if left_side else goodness[left] - change_col_penalty
            middle_choice = goodness[mid]
            right_choice = -math.inf if right_side else goodness[right] - change_col_penalty

            choices = [left_choice, middle_choice, right_choice]
            choice_index = np.argmax(choices)
            goodness[here] = choices[choice_index] + edges[here]
            parent[here] = choice_index - 1;

    seam_end = end
    if (end == None):
        seam_end = np.argmax(goodness[-1])

    '''
    print(edges)
    print(goodness)

    plot_intensity(goodness[-1])

    print(goodness[-1])
    print(edges.sum(axis=0))
    '''

    seam = [(seam_end,edges.shape[1]-1)]
    while seam[-1][1] > 0:
        seam.append((seam[-1][0] + parent[seam[-1]], seam[-1][1] - 1))


    return seam

def gaussian_kernel(size, sigma=1):
    """
    Generate a 2D Gaussian kernel.

    Parameters:
        size (int): Size of the kernel (should be odd).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: 2D Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def smart_nn_downscale(input, edges, new_width, new_height, edge_avoidance_factor = 1):
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

    Image.fromarray(chosen_point).save('..\\test\\chosen_points.png')


    return Image.fromarray(result)

def index_with_zero_padding(arr, idx):
    # Check if the provided index is within the bounds of the array dimensions
    if all(0 <= idx[dim] < arr.shape[dim] for dim in range(len(arr.shape))):
        # All indices are within bounds, return the corresponding value
        return arr[idx]
    else:
        # One or more indices are out of bounds, return 0 for zero-padding
        return 0

def multi_phase_smart_nn_downscale(input, edges, new_width, new_height, edge_avoidance_factor = 1):
    print(f"start: {time.time()}")
    image = np.array(input)
    interiors = -(edges.astype(np.float32) / 255)

    old_height, old_width, num_channels = image.shape

    result_sample_locations = np.zeros((new_height, new_width, 2), dtype=np.int32)
    result = np.zeros((new_height, new_width, num_channels), dtype=image.dtype)

    # Compute downscale factor
    height_ratio = old_height / new_height
    width_ratio = old_width / new_width

    for y in range(new_width):
        for x in range(new_height):
            nearest_neighbor_y = round((y + 0.5) * height_ratio)
            nearest_neighbor_x = round((x + 0.5) * width_ratio)
            result_sample_locations[x,y] = [nearest_neighbor_x, nearest_neighbor_y]

    chosen_point = np.copy(image)

    print(f"rec adjust start: {time.time()}")
    recursively_adjust_sample_points(result_sample_locations, interiors, search_range = 2, split_factor=4, chosen_point = chosen_point)
    print(f"rec adjust finish: {time.time()}")

    for x in range(result_sample_locations.shape[0]):
        for y in range(result_sample_locations.shape[1]):
            pixel = (x,y)
            sample_point = result_sample_locations[pixel]
            result[pixel] = image[tuple(sample_point)]

    Image.fromarray(chosen_point).save('..\\test\\chosen_points.png')
    print(f"end: {time.time()}")
    return Image.fromarray(result)


def recursively_adjust_sample_points(sample_points, interior, search_range = 3, edge_avoidance_factor=70, split_factor=2, level=0, chosen_point = None):
    colors = [[255,0,0], [210, 0, 40],  [170, 0, 80], [130, 0, 120], [90, 0, 160], [50, 0, 200], [10, 0, 240]]
    if (level == 0):
        for x in range(sample_points.shape[0]):
            for y in range(sample_points.shape[1]):
                pixel = (x, y)
                sample_point = sample_points[pixel]
                chosen_point[tuple(sample_point)] = colors[0]
    adjust_sample_points(sample_points, interior, search_range, edge_avoidance_factor)
    if level + 1 < len(colors):
        for x in range(sample_points.shape[0]):
            for y in range(sample_points.shape[1]):
                pixel = (x, y)
                sample_point = sample_points[pixel]
                chosen_point[tuple(sample_point)] = colors[level+1]


    if sample_points.shape[0] <= 1 and sample_points.shape[1] <= 1:
        return;

    row_splits = min(split_factor, sample_points.shape[0])
    col_splits = min(split_factor, sample_points.shape[1])

    for row_split in np.array_split(sample_points, row_splits, axis=0):  # for each subarray split along the 0-th (row) axis
        for submatrix in np.array_split(row_split, col_splits, axis=1):  # for each subarray split along the 1-th (column) axis
            recursively_adjust_sample_points(submatrix, interior, search_range, edge_avoidance_factor, split_factor, level+1, chosen_point)


def adjust_sample_points(sample_points, interior, search_range = 3, edge_avoidance_factor=1):
    best_offset = np.array([0, 0])
    best_offset_value = -math.inf
    for x in range(-search_range, search_range):
        for y in range(-search_range, search_range):
            offset = np.array([x, y])
            value = edge_avoidance_factor * sample_points_value(sample_points + offset, interior)
            #print(f"value: {value}")
            value -= np.linalg.norm(offset)
            #print(f"offset: {np.linalg.norm(offset)}")
            if (value > best_offset_value):
                best_offset = offset
                best_offset_value = value
    sample_points += best_offset



def sample_points_value(sample_points, interior):
    # Get the shape of the interior array
    max_row, max_col = interior.shape

    # Extract rows and columns from sample_points
    rows, cols = sample_points[:, :, 0], sample_points[:, :, 1]

    # Ignore if the choice creates out-of-bounds indices
    if (rows >= interior.shape[0]).any() or (cols >= interior.shape[1]).any() or (rows < 0).any() or (cols < 0).any():
        return -np.inf

    values = interior[rows, cols]

    # Sum the values and return
    return np.average(values)


def plot_ft(frequencies, averaged_summed_fft):
    plt.plot(frequencies, averaged_summed_fft)
    plt.xlabel('Frequency')
    plt.ylabel('Average Sum of Fourier Transform')
    plt.title('Average Fourier Transform Sum')
    plt.show()


def process_images_in_folder(folder_path, common_length=1024):
    all_summed_ffts = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file types as needed
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            edges = sobel_color(image, ksize=1)

            # Subtract the mean
            edges_mean_subtracted = edges - np.mean(edges, axis=1, keepdims=True)

            # Perform Fourier Transform on each row
            fft_rows = np.fft.fft(edges_mean_subtracted, axis=1)

            # Sum the results
            summed_fft = np.sum(np.abs(fft_rows), axis=0)

            # Normalize the length by interpolation
            original_length = len(summed_fft)
            x_original = np.linspace(0, 1, original_length)
            x_common = np.linspace(0, 1, common_length)
            interpolator = interp1d(x_original, summed_fft, kind='linear')
            normalized_fft = interpolator(x_common)

            all_summed_ffts.append(normalized_fft)

    if not all_summed_ffts:
        raise ValueError("No images processed or all files were invalid.")

    # Compute the average of all normalized FFTs
    averaged_summed_fft = np.mean(all_summed_ffts, axis=0)

    # Get the frequencies for plotting
    frequencies = np.fft.fftfreq(common_length)

    # Plot the average frequency graph
    plot_ft(frequencies, averaged_summed_fft)

if __name__ == "__main__" and False:
    folder_path = "..\\sample-non-pixel"  # Replace with the path to your folder
    process_images_in_folder(folder_path)


if __name__ == "__main__":
    input_image_path = "..\\nature.jpg"
    output_path = "..\\nature - smart nn.png"
    image = Image.open(input_image_path)


    edges = sobel_color(image, ksize=1)
    Image.fromarray(edges).save("..\\edges.png")
    scale_factor = scale_factor(edges)

    fourier_transform_and_plot(edges)

    # Calculate the new size
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    resized_image = smart_nn_downscale(image, edges, new_width, new_height, edge_avoidance_factor=1)
    resized_image.save(output_path, format="PNG")

    #palette = ColorPalette()
    #palette.generate_palette_from_image('k_means', resized_image, 22)
    #palletized_image = palette.apply_palette_to_image(resized_image)
    #palletized_image.save(output_path, format="PNG")

    '''
    edges = canny(image)

    seam = find_seam(np.rot90(edges, k=0))

    output = np.copy(np.rot90(edges, k=0))
    for point in seam:
        output[(point[1],point[0])] = 255


    cv2.imshow('Original Image', pillow_to_cv2(image))
    cv2.imshow('Edge Detection (Color)', edges)
    cv2.imshow('Seam', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
