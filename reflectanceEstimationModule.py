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
num_epochs = 10  # Adjust the number of epochs as needed
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

# Define the lightweight CNN
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

# Instantiate the model
model = CoefficientPredictor().to(device)

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

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for I_l, I_h in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        I_l = I_l.to(device)
        I_h = I_h.to(device)

        # Predict coefficients
        coefficients = model(I_l)
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

    # Save sample enhanced images
    model.eval()
    with torch.no_grad():
        sample_I_l = I_l[:4]  # Get a batch of low-light images
        sample_I_h = I_h[:4]  # Corresponding high-light images
        coefficients = model(sample_I_l)
        a, b = map_coefficients(coefficients)
        I_h_enhanced = brightness_contrast_adjustment(sample_I_l, a, b)

        # Visualize and save the images
        for idx in range(sample_I_l.size(0)):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(sample_I_l[idx].cpu().permute(1, 2, 0).numpy())
            axs[0].set_title('Low-Light Input')
            axs[0].axis('off')

            axs[1].imshow(I_h_enhanced[idx].cpu().permute(1, 2, 0).numpy())
            axs[1].set_title('Enhanced Output')
            axs[1].axis('off')

            axs[2].imshow(sample_I_h[idx].cpu().permute(1, 2, 0).numpy())
            axs[2].set_title('Ground Truth')
            axs[2].axis('off')

            plt.savefig(f'output_epoch_{epoch+1}_sample_{idx+1}.png')
            plt.close()

# Save the trained model
torch.save(model.state_dict(), 'coefficient_predictor.pth')