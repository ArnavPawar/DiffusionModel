import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diffusion model parameters
num_steps = 1000  # Number of diffusion steps
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_steps).to(device)  # Linear beta schedule
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# Helper function to add noise
def add_noise(img, t, alpha_bars):
    alpha_bar_t = alpha_bars[t]
    noise = torch.randn_like(img)
    noisy_img = torch.sqrt(alpha_bar_t) * img + torch.sqrt(1 - alpha_bar_t) * noise
    return noisy_img, noise

# Function to compute the mean of q(x_{t-1} | x_t, x_0)
def compute_posterior_mean(x_t, t, noise, alpha_bars, betas):
    alpha_bar_t = alpha_bars[t]
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=device)
    
    # Estimate x_0 from x_t and noise
    x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * noise) / torch.sqrt(alpha_bar_t)
    
    # Compute the mean of q(x_{t-1} | x_t, x_0)
    posterior_mean = (
        torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t) * x_0_hat +
        torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * x_t
    )
    return posterior_mean

# Function to load and preprocess an image
def load_image(image_path, size):
    img = Image.open(image_path).convert("RGB").resize((size, size))
    img = transforms.ToTensor()(img).unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]
    return img.to(device)

# Load and preprocess the input image
image_path = "image.jpg"  # Replace with the path to your image
img_size = 64             # Set image size (e.g., 64x64)
img = load_image(image_path, img_size)

# Display the original image
fig, axs = plt.subplots(1, 11, figsize=(22, 2))
axs[0].imshow(((img.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1))
axs[0].axis("off")
axs[0].set_title("Original")

# Add noise to the image
t = num_steps - 1
noisy_img, noise = add_noise(img, t, alpha_bars)

# Run reverse diffusion process to reconstruct the original image
reconstructed_imgs = [noisy_img.detach().cpu()]  # Store each step for visualization

for t in range(num_steps - 1, 0, -1):
    posterior_mean = compute_posterior_mean(noisy_img, t, noise, alpha_bars, betas)
    noisy_img = posterior_mean + torch.sqrt(betas[t]) * torch.randn_like(noisy_img)
    reconstructed_imgs.append(noisy_img.detach().cpu())

# Visualize the noisy and reconstructed results
for i in range(1, 11):
    step_img = reconstructed_imgs[int(i * num_steps / 10) - 1].squeeze(0).permute(1, 2, 0)
    step_img = ((step_img.numpy() + 1) / 2).clip(0, 1)  # Denormalize to [0, 1]
    axs[i].imshow(step_img)
    axs[i].axis("off")
    if i == 1:
        axs[i].set_title("Noisy")
    elif i == 10:
        axs[i].set_title("Reconstructed")

plt.show()