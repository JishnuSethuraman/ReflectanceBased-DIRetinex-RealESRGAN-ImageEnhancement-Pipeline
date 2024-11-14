# Reflectance-Based DIRetinex and Real-ESRGAN Image Enhancement Pipeline

## Introduction
This project implements an image enhancement pipeline that combines a Reflectance-Based Deep Retinex model with Real-ESRGAN for low-light image enhancement and super-resolution. The pipeline enhances low-light images by improving brightness and contrast while preserving details, then upscales the enhanced images using super-resolution techniques for improved visual quality.

## Concepts

### Reflectance-Based Deep Retinex Model
The Retinex theory models an image as a product of reflectance and illumination components:
- **Reflectance**: Represents the intrinsic properties of objects in the scene, such as color and texture.
- **Illumination**: Represents the lighting conditions affecting the scene.

This project uses the recently published DI-Retinex model, a lightweight Convolutional Neural Network (CNN) that enhances visibility by predicting brightness and contrast adjustment coefficients, achieving natural-looking results.

### Real-ESRGAN
Real-ESRGAN is a GAN-based approach for image super-resolution, enhancing image details and textures. By integrating Real-ESRGAN into the pipeline, the enhanced low-light images are upscaled, producing high-resolution outputs with improved visual quality.

## Project Overview

### Architecture
The pipeline consists of two main components:
1. **Reflectance Estimation Module**: Enhances low-light images by adjusting brightness and contrast based on predicted coefficients.
2. **Super-Resolution Module (Real-ESRGAN)**: Upscales the enhanced images, adding finer details.

### Workflow
1. **Input**: Low-light images from the LOL dataset.
2. **Reflectance Estimation**: The CNN predicts adjustment coefficients for each image.
3. **Image Enhancement**: Brightness and contrast adjustments are applied to enhance images.
4. **Super-Resolution**: Enhanced images are upscaled using Real-ESRGAN.
5. **Output**: High-resolution, enhanced images are saved and compared against ground truth high-light images.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended)
- **Python packages**: `numpy`, `matplotlib`, `pillow`, `tqdm`, `torch`, `torchvision`, `basicsr`, `realesrgan`

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/JishnuSethuraman/ReflectanceBased-DIRetinex-RealESRGAN-ImageEnhancement-Pipeline.git
   cd ReflectanceBased-DIRetinex-RealESRGAN-ImageEnhancement-Pipeline
   
2. **Create a Virtual Environment (optional)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/MacOS
   source venv/bin/activate
   
3. **Install Required Packages**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision
   pip install numpy matplotlib pillow tqdm
   pip install basicsr realesrgan

   4. **Download Real-ESRGAN Model Weights**
   - Download `RealESRGAN_x4plus.pth` from the [Real-ESRGAN release page](https://github.com/xinntao/Real-ESRGAN).
   - Create a `weights` directory in the project root and place the model file inside:
     ```bash
     mkdir weights
     mv RealESRGAN_x4plus.pth weights/
     ```

5. **Download the LOL Dataset**
   - Download the LOL dataset and place it in the project directory.
   - Ensure the dataset structure is as follows:
     ```plaintext
     LOLdataset/
       our485/
         low/
         high/
       eval15/
         low/
         high/
     ```

## Usage

1. **Running the Script**
   ```bash
   python comparison_image_enhancement_pipeline.py
The script processes 10 random images from the `our485/low` directory, generating comparison grids for:

- **Original Low-Light Image**
- **Enhanced Low-Light Image**
- **Super-Resolved Image**
- **Ground Truth High-Light Image**

Outputs are saved in the `processed_images_with_comparisons` directory.

### Adjusting the Number of Images
To process a different number of images, modify the `selected_images` line in the script:
   ```python
   selected_images = random.sample(all_images, N)  # Replace N with the desired number
```
### Processing Specific Images
Provide a list of specific filenames to process:
   ```python
   selected_images = ['1.png', '10.png', '23.png']  # Replace with actual filenames
```
