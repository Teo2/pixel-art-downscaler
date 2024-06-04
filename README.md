# Pixel Art AI Enhancer

Generating pixel art using AI models like DALL-E can be challenging. While the AI might produce an image that resembles pixel art at first glance, it's often a 1024x1024 canvas that doesn't downscale well to true pixel art. This program addresses that challenge by refining AI-generated images into authentic pixel art.

## Features

This program consists of three main parts:

### 1. Pixel Size Determination

The first step is to determine the attempted pixel size in the image, essentially guessing the downscale ratio that the AI intended. This is achieved by:
- Applying an edge detection filter to identify the boundaries of the pixels.
- Performing a Fourier transform to analyze the frequency of these pixel edges and identify the most common gap size between them.

### 2. Downscaling

Once the pixel size is determined, the image is downscaled. This step involves:
- Utilizing a method similar to nearest neighbor downscaling, which preserves the sharp edges characteristic of pixel art.
- Avoiding areas that are too noisy or do not align well with the pixel edges to maintain clarity and authenticity.

### 3. Color Compression

Pixel art typically uses a limited color palette. To achieve this:
- The program identifies the most prominent colors in the image.
- It then restricts the image to this reduced set of colors, ensuring the result adheres to the aesthetic of traditional pixel art.

## Examples

Here is an example of what the program can do:




This program bridges the gap between AI-generated images and true pixel art, ensuring a high-quality, authentic output.
