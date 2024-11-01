import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple diffusion model parameters
num_steps = 1000  # Number of diffusion steps
img_size = 64     # Image size (e.g., 64x64)
beta = 0.02       # Noise addition coefficient

# Helper function to add noise
def add_noise(img, t, beta):
    noise = torch.randn_like(img) * (beta * t)
    return img + noise

# Helper function to reverse noise at each step
def reverse_diffusion_step(img, t, beta):
    noise = torch.randn_like(img) * (beta * (num_steps - t))
    return img - noise

# Create a sample image (a simple gradient here)
def create_sample_image(size):
    img = np.linspace(0, 1, size * size).reshape(size, size)
    img = np.stack([img, img, img], axis=2)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

# Initialize sample image and noisy version
img = create_sample_image(img_size)
noisy_img = add_noise(img, num_steps, beta)

# Run reverse diffusion process
reconstructed_imgs = [noisy_img]  # Store each step for visualization

for t in range(num_steps - 1, -1, -1):
    noisy_img = reverse_diffusion_step(noisy_img, t, beta)
    reconstructed_imgs.append(noisy_img.detach().cpu())

# Visualize the results
fig, axs = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axs):
    step_img = reconstructed_imgs[int(i * num_steps / 10)].squeeze(0).permute(1, 2, 0).numpy()
    ax.imshow(step_img, cmap="gray")
    ax.axis("off")
plt.show()
