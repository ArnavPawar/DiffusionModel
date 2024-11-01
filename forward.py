import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess the image
def load_image(image_path, size=(128, 128)):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# Define the forward diffusion process
def forward_diffusion(x, num_steps, beta_start=0.0001, beta_end=0.02):
    # Linearly spaced noise levels (betas)
    betas = torch.linspace(beta_start, beta_end, num_steps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    noisy_images = []
    for t in range(num_steps):
        # Sample noise
        noise = torch.randn_like(x)
        # Compute the noisy image at step t
        noisy_image = torch.sqrt(alphas_cumprod[t]) * x + torch.sqrt(1 - alphas_cumprod[t]) * noise
        noisy_images.append(noisy_image)
    
    return noisy_images

# Visualize the diffusion steps
def visualize_diffusion(noisy_images, steps_to_show):
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(15, 5))
    for i, step in enumerate(steps_to_show):
        img = noisy_images[step].squeeze().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # Ensure pixel values are in [0, 1] range
        axes[i].imshow(img)
        axes[i].set_title(f"Step {step}")
        axes[i].axis('off')
    plt.show()

# Set parameters and run the diffusion
image_path = "image.jpg"  # Specify your image path here
image = load_image(image_path)
num_steps = 50  # Number of forward diffusion steps
noisy_images = forward_diffusion(image, num_steps)

# Visualize selected steps
steps_to_show = [0, 10, 20, 30, 40, 49]
visualize_diffusion(noisy_images, steps_to_show)
