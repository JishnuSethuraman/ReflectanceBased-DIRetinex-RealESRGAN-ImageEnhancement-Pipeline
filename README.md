# Reflectance-Based DIRetinex and Real-ESRGAN Image Enhancement Pipeline

## Introduction
This project implements an image enhancement pipeline that combines a Reflectance-Based Deep Retinex model with Real-ESRGAN for low-light image enhancement and super-resolution. The pipeline enhances low-light images by improving brightness and contrast while preserving details, then upscales the enhanced images using super-resolution techniques for improved visual quality.

## Concepts

### Reflectance-Based Deep Retinex Model
The Retinex theory models an image as a product of reflectance and illumination components:
- **Reflectance**: Represents the intrinsic properties of objects in the scene, such as color and texture.
- **Illumination**: Represents the lighting conditions affecting the scene.

This project uses the recently published [DI-Retinex model](https://arxiv.org/abs/2404.03327), a lightweight Convolutional Neural Network (CNN) that enhances visibility by predicting brightness and contrast adjustment coefficients, achieving natural-looking results.

### Real-ESRGAN
[Real-ESRGAN](https://github.com/xinntao/ESRGAN) is a GAN-based approach for image super-resolution, enhancing image details and textures. By integrating Real-ESRGAN into the pipeline, the enhanced low-light images are upscaled, producing high-resolution outputs with improved visual quality.

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
## Code Structure

The project repository is organized as follows:

- **`LOLdataset/`**: Contains the low-light and high-light images from the LOL dataset.
- **`processed_images/`**: Directory for intermediate processed images.
- **`processed_images_with_comparisons/`**: Contains the final comparison images generated by the pipeline.
- **`weights/`**: Contains model weights, including:
  - **`coefficient_predictor.pth`**: Pretrained weights for the coefficient predictor model.
  - **`net.pth`**: Additional model weights.
  - **`srcnn_x2.pth`**: Pretrained weights for super-resolution (if applicable).
- **`.gitattributes`**: Git configuration file.
- **`LICENSE`**: License file for the project.
- **`README.md`**: The project README file.
- **`comparison_image_enhancement_pipeline.py`**: Main script that runs the pipeline and generates comparison figures.
- **`final_output_image.jpg`**: Example of the final output image generated by the pipeline.
- **`test_imports.py`**: Script to test imports and dependencies.

### Scripts and Modules

- **`comparison_image_enhancement_pipeline.py`**: The main script to execute the image enhancement pipeline.
  - **Functions**:
    - `map_coefficients`: Maps raw coefficients to meaningful adjustment values.
    - `brightness_contrast_adjustment`: Applies brightness and contrast adjustments to enhance images.
    - `initialize_realesrgan_model`: Loads the Real-ESRGAN model.
    - `process_image_with_comparison`: Processes individual images and generates comparison grids.
- **`test_imports.py`**: A script to verify that all necessary modules and packages are correctly installed.

### Model Weights

- **`coefficient_predictor.pth`**: Pretrained weights for the coefficient predictor model used in reflectance estimation.
- **`net.pth`**: Model weights for the reflectance estimation module.
- **`srcnn_x2.pth`**: Pretrained weights for the SRCNN model used in super-resolution (if applicable).

### Directories

- **`LOLdataset/`**: Dataset directory containing:
  - `our485/`: Training image pairs.
    - `low/`: Low-light images.
    - `high/`: Corresponding high-light images.
  - `eval15/`: Evaluation image pairs.
    - `low/`: Low-light images.
    - `high/`: Corresponding high-light images.
- **`processed_images/`**: Stores images after enhancement but before super-resolution.
- **`processed_images_with_comparisons/`**: Stores the final images with comparison grids showing all stages.
- **`weights/`**: Contains all the necessary model weights for the pipeline.

## Dataset

- **LOL Dataset**: Used for training and evaluating low-light image enhancement algorithms. Contains pairs of low-light and corresponding high-light images.
  - **`our485`**: Contains 485 training image pairs.
  - **`eval15`**: Contains 15 evaluation image pairs.

## Results

The output comparison figures allow you to visually assess the performance of the enhancement pipeline. Each figure shows:

1. **Original Low-Light Image**: The raw input image with poor visibility.
2. **Enhanced Low-Light Image**: The image after brightness and contrast enhancement.
3. **Super-Resolved Image**: The enhanced image upscaled with added details.
4. **Ground Truth High-Light Image**: The reference image with proper lighting.

These comparison images are saved in the `processed_images_with_comparisons/` directory.

## Contributing

Contributions are welcome! If you'd like to improve the pipeline or add new features:

1. **Fork the repository.**
2. **Create a new branch for your feature:**
   ```bash
   git checkout -b feature/YourFeatureName

3. **Commit your changes with descriptive messages:**
   ```bash
   git commit -am "Add new feature: YourFeatureName"
   
4. **Push to the branch:**
   ```bash
   git push origin feature/YourFeatureName

5. **Submit a pull request** explaining your changes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Real-ESRGAN**: Xintao Wang and contributors for the Real-ESRGAN implementation.
- **LOL Dataset**: Wei Wang and colleagues for providing the dataset.
- **PyTorch Community**: For the development of deep learning frameworks and tools.

