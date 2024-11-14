# File: comparison_image_enhancement_pipeline.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import Real-ESRGAN components
from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Parameters
num_epochs = 5  # Adjust the number of epochs as needed
batch_size = 8
learning_rate = 1e-4
gamma = 2.2  # Gamma value for gamma correction

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
        low_img_name = self.low_light_images[idx]
        high_img_name = self.high_light_images[idx]

        low_img_path = os.path.join(self.low_light_dir, low_img_name)
        high_img_path = os.path.join(self.high_light_dir, high_img_name)

        try:
            low_img = Image.open(low_img_path).convert('RGB')
            high_img = Image.open(high_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {low_img_path}, {high_img_path}")
            raise e

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img, low_img_name  # Return filename

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
        for I_l, I_h, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
# Real-ESRGAN Model Integration
# =========================

# Initialize the Real-ESRGAN model
def initialize_realesrgan_model(model_name='RealESRGAN_x4plus.pth'):
    model_path = os.path.join('weights', model_name)
    if not os.path.exists(model_path):
        print(f"Model weights '{model_name}' not found in 'weights/' directory.")
        print("Please download the Real-ESRGAN model weights and place them in the 'weights/' directory.")
        exit()

    # Initialize the RRDBNet model
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )

    # Initialize the RealESRGANer
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=not device.type == 'cpu'  # Use FP16 only on GPU
    )

    return upsampler

# Instantiate the Real-ESRGAN model
realesrgan_model = initialize_realesrgan_model()
print("Real-ESRGAN model loaded successfully.")

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

# Function to perform super-resolution using Real-ESRGAN
def super_resolve_image(img_np):
    # img_np: NumPy array in BGR format
    output, _ = realesrgan_model.enhance(img_np)
    return output  # Output is a NumPy array in BGR format

# Function to process image through all steps and create comparison
def process_image_with_comparison(low_img_path, high_img_path, output_dir):
    img_name = os.path.basename(low_img_path)
    output_image_path = os.path.join(output_dir, f"processed_{img_name}")
    comparison_image_path = os.path.join(output_dir, f"comparison_{os.path.splitext(img_name)[0]}.png")

    # Load and preprocess the low-light image
    low_img = Image.open(low_img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(low_img).unsqueeze(0).to(device)

    # Enhance low-light image
    enhanced_tensor = enhance_low_light_image(img_tensor)
    enhanced_img = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu())

    # Convert enhanced image to NumPy array (BGR format)
    enhanced_img_np = np.array(enhanced_img)[:, :, ::-1]  # Convert RGB to BGR

    # Super-resolve the enhanced image using Real-ESRGAN
    sr_img_np = super_resolve_image(enhanced_img_np)

    # Convert the super-resolved image back to PIL Image (RGB)
    sr_img = Image.fromarray(sr_img_np[:, :, ::-1])  # Convert BGR to RGB

    # Save the final output image
    sr_img.save(output_image_path)
    print(f"Processed image saved at {output_image_path}")

    # Load the ground truth high-light image
    high_img = Image.open(high_img_path).convert("RGB")

    # Create comparison figure
    create_comparison_figure(low_img, enhanced_img, sr_img, high_img, comparison_image_path)

# Function to create and save comparison figure
def create_comparison_figure(low_img, enhanced_img, sr_img, high_img, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    axs[0, 0].imshow(low_img)
    axs[0, 0].set_title('Original Low-Light Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(enhanced_img)
    axs[0, 1].set_title('Enhanced Low-Light Image')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(sr_img)
    axs[1, 0].set_title('Super-Resolved Image')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(high_img)
    axs[1, 1].set_title('Ground Truth High-Light Image')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison figure saved at {save_path}")

# Function to process a batch of images with comparisons
def process_image_batch_with_comparisons(image_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in image_names:
        low_img_path = os.path.join(low_light_image_dir, img_name)
        high_img_path = os.path.join(high_light_image_dir, img_name)

        if not os.path.exists(low_img_path) or not os.path.exists(high_img_path):
            print(f"Skipping {img_name} as corresponding images not found.")
            continue

        print(f"Processing {img_name}...")
        process_image_with_comparison(low_img_path, high_img_path, output_dir)

# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    # Select 10 random images from the low-light image directory
    all_images = [fname for fname in os.listdir(low_light_image_dir) if fname.lower().endswith(IMAGE_EXTENSIONS)]
    selected_images = random.sample(all_images, 10)

    # Output directory
    output_dir = 'processed_images_with_comparisons'

    # Process the batch of images with comparisons
    process_image_batch_with_comparisons(selected_images, output_dir)