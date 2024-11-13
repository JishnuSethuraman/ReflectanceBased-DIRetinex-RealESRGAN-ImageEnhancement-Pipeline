# File: image_enhancement_pipeline.py

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
num_epochs = 5  # Adjust the number of epochs as needed
batch_size = 8
learning_rate = 1e-4
gamma = 2.2  # Gamma value for gamma correction
k = 255  # Maximum pixel value
tau = 0.01  # Small constant for mapping function

# Paths to the LOL dataset (adjust these paths accordingly)
low_light_image_dir = 'LOLdataset/our485/low'
high_light_image_dir = 'LOLdataset/our485/high'

# Ensure the directories exist
assert os.path.exists(low_light_image_dir), "Low-light image directory not found."
assert os.path.exists(high_light_image_dir), "High-light image directory not found."

# Allowed image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Custom dataset class with file filtering
class LOLDataset(Dataset):
    def __init__(self, low_light_dir, high_light_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.high_light_dir = high_light_dir
        self.transform = transform

        # Filter out non-image files
        self.low_light_images = sorted([
            fname for fname in os.listdir(self.low_light_dir)
            if fname.lower().endswith(IMAGE_EXTENSIONS)
        ])
        self.high_light_images = sorted([
            fname for fname in os.listdir(self.high_light_dir)
            if fname.lower().endswith(IMAGE_EXTENSIONS)
        ])

        assert len(self.low_light_images) == len(self.high_light_images), \
            f"Mismatch in number of images: {len(self.low_light_images)} low-light images, {len(self.high_light_images)} high-light images."

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        high_img_path = os.path.join(self.high_light_dir, self.high_light_images[idx])

        try:
            low_img = Image.open(low_img_path).convert('RGB')
            high_img = Image.open(high_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {low_img_path}, {high_img_path}")
            raise e

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img

# Define transforms
transform = transforms.Compose([
    transforms.Resize((400, 600)),  # Resize images for faster training
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = LOLDataset(low_light_image_dir, high_light_image_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the lightweight CNN for Reflectance Estimation
class CoefficientPredictor(nn.Module):
    def __init__(self):
        super(CoefficientPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, padding=1)  # Output channels for b and c
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        coefficients = torch.tanh(self.conv3(out))  # Output in [-1, 1]
        return coefficients

# Instantiate the reflectance estimation model
reflectance_model = CoefficientPredictor().to(device)

# Define the mapping function for 'a' and 'b'
def map_coefficients(coefficients):
    b_raw = coefficients[:, 0:1, :, :]  # Brightness coefficient
    c_raw = coefficients[:, 1:2, :, :]  # Contrast coefficient

    # Map 'b' from [-1, 1] to [-0.3, 0.3]
    b = b_raw * 0.3

    # Map 'a' from [-1, 1] to [0.1, 10.0]
    a_min, a_max = 0.1, 10.0
    log_ratio = torch.log(torch.tensor(a_max / a_min, device=coefficients.device))
    a = a_min * torch.exp((c_raw + 1) * log_ratio / 2)
    return a, b

# Simplify brightness and contrast adjustment
def brightness_contrast_adjustment(I_l, a, b):
    adjusted = a * I_l + b
    adjusted = torch.clamp(adjusted, 0, 1)
    return adjusted

# Ensure numerical stability in reverse degradation loss
def reverse_degradation_loss(I_l, I_h_enhanced, gamma=2.2, eta=0.5, sigma=0.1, epsilon=1e-6):
    # Apply gamma correction
    I_l_gamma = torch.pow(I_l + epsilon, 1 / gamma)
    I_h_enhanced_gamma = torch.pow(I_h_enhanced + epsilon, 1 / gamma)

    # Compute the exposure factor 'r_prime'
    E = torch.abs(torch.normal(mean=eta, std=sigma, size=(I_l.size(0), 1, 1, 1)).to(device))
    E = torch.clamp(E, min=epsilon)
    mean_I_l = I_l.mean(dim=[1,2,3], keepdim=True) + epsilon
    r_prime = (mean_I_l / E) ** (1 / gamma)

    # Compute the loss with masking
    mask = (I_h_enhanced < 1.0).float()
    loss = torch.mean(mask * torch.abs(I_l_gamma - r_prime * I_h_enhanced_gamma))
    return loss

# Define the variance suppression loss
def variance_suppression_loss(a, b):
    term = a * b - a + b + 1
    variance = torch.var(term, dim=[2, 3], unbiased=False)
    loss = torch.mean(variance)
    return loss

# Define the total loss function
def total_loss(I_l, I_h_enhanced, a, b):
    rd_loss = reverse_degradation_loss(I_l, I_h_enhanced, gamma)
    vs_loss = variance_suppression_loss(a, b)
    return rd_loss + vs_loss

# Define the optimizer for reflectance estimation model
optimizer = torch.optim.Adam(reflectance_model.parameters(), lr=learning_rate)

# Check if the trained model exists
model_path = 'coefficient_predictor.pth'
if os.path.exists(model_path):
    # Load the trained model
    reflectance_model.load_state_dict(torch.load(model_path, map_location=device))
    print("Reflectance Estimation Model loaded successfully.")
else:
    # Train the model
    print("Training Reflectance Estimation Model...")
    for epoch in range(num_epochs):
        reflectance_model.train()
        epoch_loss = 0
        for I_l, I_h in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            I_l = I_l.to(device)
            I_h = I_h.to(device)

            # Predict coefficients
            coefficients = reflectance_model(I_l)
            a, b = map_coefficients(coefficients)

            # Adjust brightness and contrast
            I_h_enhanced = brightness_contrast_adjustment(I_l, a, b)

            # Compute loss
            loss = total_loss(I_l, I_h_enhanced, a, b)

            # Check for NaNs
            if torch.isnan(loss):
                print("Loss is NaN, stopping training.")
                break

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(reflectance_model.state_dict(), model_path)
    print("Reflectance Estimation Model trained and saved.")

# =========================
# SRCNN Model
# =========================

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load the SRCNN model
srcnn_model = SRCNN().to(device)

# Load pretrained weights if available
try:
    srcnn_model.load_state_dict(torch.load('srcnn.pth', map_location=device))
    print("Pretrained SRCNN weights loaded successfully.")
except FileNotFoundError:
    print("No pretrained SRCNN weights found. Please download 'srcnn.pth' and place it in the current directory.")
    print("You can download pretrained SRCNN weights from: https://github.com/yjn870/SRCNN-pytorch")

# Set the SRCNN model to evaluation mode
srcnn_model.eval()

# =========================
# DnCNN Model for Noise Reduction
# =========================

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning

# Load the DnCNN model
dncnn_model = DnCNN().to(device)

# Load pretrained weights for DnCNN
try:
    dncnn_model.load_state_dict(torch.load('dncnn.pth', map_location=device))
    print("Pretrained DnCNN weights loaded successfully.")
except FileNotFoundError:
    print("No pretrained DnCNN weights found. Please download 'dncnn.pth' and place it in the current directory.")
    print("You can download pretrained DnCNN weights from: https://github.com/SaoYan/DnCNN-PyTorch")

# Set the DnCNN model to evaluation mode
dncnn_model.eval()

# =========================
# Image Processing Functions
# =========================

# Function to enhance low-light image
def enhance_low_light_image(img_tensor):
    with torch.no_grad():
        coefficients = reflectance_model(img_tensor)
        a, b = map_coefficients(coefficients)
        enhanced_tensor = brightness_contrast_adjustment(img_tensor, a, b)
    return enhanced_tensor

# Function to perform super-resolution using SRCNN
def super_resolve_image(img_tensor, upscale_factor=2):
    with torch.no_grad():
        # Upsample the image using bicubic interpolation
        upscaled_tensor = nn.functional.interpolate(img_tensor, scale_factor=upscale_factor, mode='bicubic', align_corners=False)
        # Apply SRCNN
        sr_tensor = srcnn_model(upscaled_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
    return sr_tensor

# Function to apply noise reduction using DnCNN
def reduce_noise(img_tensor):
    with torch.no_grad():
        denoised_tensor = dncnn_model(img_tensor)
        denoised_tensor = torch.clamp(denoised_tensor, 0, 1)
    return denoised_tensor

# Function to process image through all steps
def process_image(input_image_path, output_image_path, upscale_factor=2):
    # Load and preprocess the image
    img = Image.open(input_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Enhance low-light image
    enhanced_tensor = enhance_low_light_image(img_tensor)

    # Super-resolve the enhanced image
    sr_tensor = super_resolve_image(enhanced_tensor, upscale_factor=upscale_factor)

    # Apply noise reduction
    final_tensor = reduce_noise(sr_tensor)

    # Save the final output image
    output_img = transforms.ToPILImage()(final_tensor.squeeze(0).cpu())
    output_img.save(output_image_path)
    print(f"Processed image saved at {output_image_path}")

    # Display the images
    display_images(img_tensor, enhanced_tensor, sr_tensor, final_tensor)

# Function to display images at different stages
def display_images(original_tensor, enhanced_tensor, sr_tensor, final_tensor):
    original_img = transforms.ToPILImage()(original_tensor.squeeze(0).cpu())
    enhanced_img = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu())
    sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    final_img = transforms.ToPILImage()(final_tensor.squeeze(0).cpu())

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(original_img)
    axs[0].set_title('Original Low-Light Image')
    axs[0].axis('off')

    axs[1].imshow(enhanced_img)
    axs[1].set_title('Enhanced Image')
    axs[1].axis('off')

    axs[2].imshow(sr_img)
    axs[2].set_title('Super-Resolved Image')
    axs[2].axis('off')

    axs[3].imshow(final_img)
    axs[3].set_title('Final Output (Denoised)')
    axs[3].axis('off')

    plt.show()

# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    # Example input image path (you can change this to your own image)
    input_image_path = 'low_light_image.jpg'  # Path to the input low-light image
    output_image_path = 'final_output_image.jpg'  # Path to save the final processed image

    if not os.path.exists(input_image_path):
        print(f"Input image '{input_image_path}' not found. Please provide a valid image path.")
    else:
        process_image(input_image_path, output_image_path, upscale_factor=2)