import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple diffusion model parameters
num_steps = 1000  # Number of diffusion steps
beta = 0.02       # Noise addition coefficient

# Helper function to add noise (simulates blurring)
def add_noise(img, t, beta):
    noise = torch.randn_like(img) * (beta * t)
    return img + noise

# Helper function to reverse noise at each step
def reverse_diffusion_step(img, t, beta):
    noise = torch.randn_like(img) * (beta * (num_steps - t))
    return img - noise

# Function to load and preprocess an image
def load_image(image_path, size):
    img = Image.open(image_path).convert("RGB").resize((size, size))
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img.to(device)

# Load and preprocess the input image
image_path = "image.jpg"  # Replace with the path to your image
img_size = 64                  # Set image size (e.g., 64x64)
img = load_image(image_path, img_size)

# Display the original image
fig, axs = plt.subplots(1, 11, figsize=(22, 2))
axs[0].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
axs[0].axis("off")
axs[0].set_title("Original")

# Add noise to the image
noisy_img = add_noise(img, num_steps, beta)

# Run reverse diffusion process to reconstruct the original image
reconstructed_imgs = [noisy_img]  # Store each step for visualization

for t in range(num_steps - 1, -1, -1):
    noisy_img = reverse_diffusion_step(noisy_img, t, beta)
    reconstructed_imgs.append(noisy_img.detach().cpu())

# Visualize the noisy and reconstructed results
for i in range(1, 11):
    step_img = reconstructed_imgs[int(i * num_steps / 10)].squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    axs[i].imshow(step_img)
    axs[i].axis("off")
    if i == 1:
        axs[i].set_title("Noisy")
    elif i == 10:
        axs[i].set_title("Reconstructed")

plt.show()