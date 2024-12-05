# Install necessary libraries

# Imports
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from diffusers import DDPMScheduler, UNet2DModel
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation: FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.FashionMNIST(
    root="fashion-mnist/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Display a sample batch
x, y = next(iter(train_dataloader))
plt.imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
plt.title("Original Sample Batch")
plt.show()

# Corruption process: Adds noise to data
def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`."""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Adjust shape for broadcasting
    return x * (1 - amount) + noise * amount

# Test corruption
amount = torch.linspace(0, 1, x.shape[0])
noisy_x = corrupt(x, amount)

# Display noisy images
fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title("Input data")
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
axs[1].set_title("Corrupted data (increasing noise)")
axs[1].imshow(torchvision.utils.make_grid(noisy_x)[0], cmap="Greys")
plt.show()

# Define a basic UNet model
class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            if i < 2:
                h.append(x)
                x = self.downscale(x)
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()
            x = self.act(layer(x))
        return x

# Initialize UNet model
net = BasicUNet().to(device)

# Loss function and optimizer
loss_fn = nn.MSELoss()
#loss_fn = nn.Smooth1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Training the diffusion model
n_epochs = 3
batch_size = 8
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
losses = []

## look at noise and not nosied image


for epoch in range(n_epochs):
    for x, _ in train_dataloader:
        x = x.to(device)
        noise_amount = torch.rand(x.shape[0], device=device)
        noisy_x = corrupt(x, noise_amount)
        pred = net(noisy_x)
        loss = loss_fn(pred, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

# Plot training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Reverse process function: Iterative denoising
def reverse_process(model, noisy_x, steps):
    """Perform reverse diffusion process to reconstruct the original image."""
    x = noisy_x.clone().to(device)
    for i in range(steps):
        with torch.no_grad():
            pred = model(x)  # Predict the clean image
        mix_factor = 1 / (steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor  # Gradually denoise
    return x

# Perform the reverse process
steps = 10
reconstructed_images = reverse_process(net, noisy_x, steps)

# Plot all stages: Original, Noisy, and Reconstructed
fig, axs = plt.subplots(3, 1, figsize=(12, 9))

# Original images
axs[0].imshow(torchvision.utils.make_grid(x)[0].cpu(), cmap="Greys")
axs[0].set_title("Original Images")

# Noisy images
axs[1].imshow(torchvision.utils.make_grid(noisy_x)[0].cpu(), cmap="Greys")
axs[1].set_title("Noisy Images (after corruption)")

# Reconstructed images
axs[2].imshow(torchvision.utils.make_grid(reconstructed_images)[0].cpu().detach(), cmap="Greys")
axs[2].set_title("Reconstructed Images (after reverse process)")

plt.tight_layout()
plt.show()