import torch
import torch.nn as nn
# Define the ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(8, out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(8, out_channels)

        # Add a 1x1 convolution for channel alignment if needed
        if in_channels != out_channels:
            self.channel_match = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_match = None

    def forward(self, x):
        residual = x
        if self.channel_match is not None:
            residual = self.channel_match(residual)
        x = self.conv1(x)
        x = self.group_norm1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.group_norm2(x)
        return x + residual


# # Define a test function
# def test_resnet_block():
#     # Parameters for testing
#     batch_size = 4
#     in_channels = 16
#     out_channels = 16
#     height = 32
#     width = 32

#     # Create a random input tensor
#     input_tensor = torch.randn(batch_size, in_channels, height, width)

#     # Instantiate the ResNetBlock
#     resnet_block = ResNetBlock(in_channels, out_channels)

#     # Pass the input through the block
#     output_tensor = resnet_block(input_tensor)

#     # Print the shapes
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output_tensor.shape)

#     # Assert output shape matches input shape
#     assert input_tensor.shape == output_tensor.shape, "Output shape does not match input shape!"

#     # If no errors, the test passes
#     print("ResNetBlock test passed!")
# # Run the test
# if __name__ == "__main__":
#     test_resnet_block()

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(DownsampleBlock, self).__init__()
        
        # MaxPool2D for downsampling
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Two ResNet Blocks
        self.resnet1 = ResNetBlock(in_channels, out_channels)
        self.resnet2 = ResNetBlock(out_channels, out_channels)
        
        # Embedding addition: Linear layer to match feature map dimensions
        self.embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels)
        )
    
    def forward(self, x, embed):
        """
        Args:
            x: Input feature map (B, C, H, W)
            embed: Embedding vector (B, embed_dim)
        Returns:
            Downsampled and processed feature map (B, C_out, H/2, W/2)
        """
        # Downsample the input
        x = self.downsample(x)
        
        # First ResNet block
        x = self.resnet1(x)
        
        # Second ResNet block
        x = self.resnet2(x)
        
        # Process the embedding and add to the feature map
        embed = self.embed_proj(embed)  # (B, out_channels)
        embed = embed.unsqueeze(-1).unsqueeze(-1)  # Reshape to (B, C, 1, 1)
        x = x + embed  # Add embedding to the feature map
        
        return x
    
# def test_downsample_block():
#     # Parameters for testing
#     batch_size = 4
#     in_channels = 16
#     out_channels = 32
#     height = 64
#     width = 64
#     embed_dim = 128

#     # Create random input tensor and embedding
#     input_tensor = torch.randn(batch_size, in_channels, height, width)
#     embed_vector = torch.randn(batch_size, embed_dim)

#     # Instantiate the Downsample Block
#     downsample_block = DownsampleBlock(in_channels, out_channels, embed_dim)

#     # Pass input through the Downsample Block
#     output_tensor = downsample_block(input_tensor, embed_vector)

#     # Print the shapes
#     print("Input shape:", input_tensor.shape)
#     print("Embedding shape:", embed_vector.shape)
#     print("Output shape:", output_tensor.shape)

#     # Assert output spatial dimensions are halved
#     assert output_tensor.shape == (batch_size, out_channels, height // 2, width // 2), \
#         "Output shape is incorrect!"

#     print("DownsampleBlock test passed!")

# # Run the test
# if __name__ == "__main__":
#     test_downsample_block()

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Normalization layers
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor with the same shape
        """
        # Reshape input for attention: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Apply LayerNorm and Multi-Head Attention
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output  # Residual connection

        # Apply LayerNorm and Feedforward
        x_norm = self.layer_norm2(x)
        x_ff = self.feedforward(x_norm)
        x = x + x_ff  # Residual connection

        # Reshape back to original dimensions: (B, H*W, C) -> (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x
    
# def test_self_attention_block():
#     # Parameters for testing
#     batch_size = 4
#     channels = 64
#     height = 32
#     width = 32
#     num_heads = 4

#     # Create a random input tensor
#     input_tensor = torch.randn(batch_size, channels, height, width)

#     # Instantiate the Self-Attention Block
#     self_attention_block = SelfAttentionBlock(embed_dim=channels, num_heads=num_heads)

#     # Pass the input through the Self-Attention Block
#     output_tensor = self_attention_block(input_tensor)

#     # Print the shapes
#     print("Input shape:", input_tensor.shape)
#     print("Output shape:", output_tensor.shape)

#     # Assert output shape matches input shape
#     assert input_tensor.shape == output_tensor.shape, "Output shape does not match input shape!"

#     print("SelfAttentionBlock test passed!")

# # Run the test
# if __name__ == "__main__":
#     test_self_attention_block()
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, skip_channels):
        super(UpsampleBlock, self).__init__()
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # First ResNet Block after upsampling
        self.resnet1 = ResNetBlock(in_channels, out_channels)
        
        # Adjust channel dimensions after concatenation
        self.channel_match = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        
        # Second ResNet Block
        self.resnet2 = ResNetBlock(out_channels, out_channels)
        
        # Embedding addition: Linear layer to match feature map dimensions
        self.embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels)
        )

    def forward(self, x, skip, embed):
        """
        Args:
            x: Input feature map from the previous layer (B, C, H, W)
            skip: Skip connection feature map from the encoder (B, C_skip, H_skip, W_skip)
            embed: Embedding vector (B, embed_dim)
        Returns:
            Output feature map with upsampled dimensions (B, C_out, 2*H, 2*W)
        """
        # Upsample the input feature map
        x = self.upsample(x)  # Double the spatial dimensions
        
        # Crop or resize the skip connection to match x
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Process with the first ResNet block
        x = self.resnet1(x)
        
        # Concatenate the skip connection
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension
        
        # Adjust channel dimensions
        x = self.channel_match(x)
        
        # Process with the second ResNet block
        x = self.resnet2(x)
        
        # Add embedding information
        embed = self.embed_proj(embed)  # (B, out_channels)
        embed = embed.unsqueeze(-1).unsqueeze(-1)  # Reshape to (B, C, 1, 1)
        x = x + embed  # Add embedding to the feature map
        
        return x



# def test_upsample_block():
#     # Parameters for testing
#     batch_size = 4
#     in_channels = 64
#     out_channels = 32
#     height = 32
#     width = 32
#     embed_dim = 128

#     # Create random input tensors
#     input_tensor = torch.randn(batch_size, in_channels, height, width)
#     skip_tensor = torch.randn(batch_size, out_channels, height * 2, width * 2)
#     embed_vector = torch.randn(batch_size, embed_dim)

#     # Instantiate the Upsample Block
#     upsample_block = UpsampleBlock(in_channels, out_channels, embed_dim)

#     # Pass input and skip connection through the Upsample Block
#     output_tensor = upsample_block(input_tensor, skip_tensor, embed_vector)

#     # Print the shapes
#     print("Input shape:", input_tensor.shape)
#     print("Skip connection shape:", skip_tensor.shape)
#     print("Embedding shape:", embed_vector.shape)
#     print("Output shape:", output_tensor.shape)

#     # Assert output shape matches expected dimensions
#     assert output_tensor.shape == (batch_size, out_channels, height * 2, width * 2), \
#         "Output shape is incorrect!"

#     print("UpsampleBlock test passed!")

# # Run the test
# if __name__ == "__main__":
#     test_upsample_block()
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, embed_dim, timesteps=1000):
        super(UNet, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = DownsampleBlock(64, 128, embed_dim)
        self.down2 = DownsampleBlock(128, 256, embed_dim)
        self.down3 = DownsampleBlock(256, 512, embed_dim)
        
        # Bottleneck with self-attention
        self.attention = SelfAttentionBlock(embed_dim=512, num_heads=4)
        
        # Upsampling path
        self.up3 = UpsampleBlock(512, 256, embed_dim, skip_channels=512)
        self.up2 = UpsampleBlock(256, 128, embed_dim, skip_channels=256)
        self.up1 = UpsampleBlock(128, 64, embed_dim, skip_channels=128)
        
        # Final output convolution
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        
        # Time embedding
        self.time_embed_proj = nn.Sequential(
            nn.Linear(timesteps, embed_dim),  # Projects (batch_size, timesteps) to (batch_size, embed_dim)
            nn.SiLU()
        )

    def forward(self, x, embed):
        # Ensure `embed` has shape (batch_size, timesteps) before passing to `time_embed_proj`
        embed = self.time_embed_proj(embed)  # Project timestep embedding to (batch_size, embed_dim)
        
        # Initial convolution
        x1 = self.initial_conv(x)
        
        # Downsampling
        d1 = self.down1(x1, embed)
        d2 = self.down2(d1, embed)
        d3 = self.down3(d2, embed)
        
        # Bottleneck with self-attention
        b = self.attention(d3)
        
        # Upsampling
        u3 = self.up3(b, d3, embed)
        u2 = self.up2(u3, d2, embed)
        u1 = self.up1(u2, d1, embed)
        
        # Final output
        out = self.final_conv(u1)
        return out


# def test_unet():
#     # Parameters for testing
#     batch_size = 4
#     input_channels = 3
#     output_channels = 3
#     height = 64
#     width = 64
#     embed_dim = 128

#     # Create random input tensors
#     input_tensor = torch.randn(batch_size, input_channels, height, width)
#     embed_vector = torch.randn(batch_size, embed_dim)

#     # Instantiate the U-Net
#     unet_model = UNet(input_channels, output_channels, embed_dim)

#     # Pass input through the U-Net
#     output_tensor = unet_model(input_tensor, embed_vector)

#     # Print the shapes
#     print("Input shape:", input_tensor.shape)
#     print("Embedding shape:", embed_vector.shape)
#     print("Output shape:", output_tensor.shape)

#     # Assert output shape matches input spatial dimensions
#     assert output_tensor.shape == (batch_size, output_channels, height, width), \
#         "Output shape is incorrect!"

#     print("UNet test passed!")

# # Run the test
# if __name__ == "__main__":
#     test_unet()

#defines noice schedule for timestamps
def noise_schedule(timesteps, s=0.008):
    x = torch.linspace(0, timesteps - 1, timesteps) / timesteps
    return torch.cos((x + s) / (1 + s) * torch.pi / 2) ** 2

#forward 
def forward_diffusion(x_0, timesteps, noise_schedule):
    """
    Args:
        x_0: Original image (B, C, H, W)
        timesteps: Number of timesteps in the diffusion process
        noise_schedule: Noise schedule (array of β_t values)
    Returns:
        A tuple (noisy_image, noise) for each timestep
    """
    noise = torch.randn_like(x_0)  # Sample random noise
    alpha = 1.0 - noise_schedule
    alpha_bar = torch.cumprod(alpha, dim=0)  # Compute cumulative product of (1 - β_t)
    
    # Sample noisy image at a random timestep
    t = torch.randint(0, timesteps, (x_0.shape[0],)).to(x_0.device)
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
    
    noisy_image = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    return noisy_image, noise, t

#loss
def diffusion_loss(model, x_0, timesteps, noise_schedule, embed_dim):
    """
    Args:
        model: U-Net model
        x_0: Original image (B, C, H, W)
        timesteps: Number of timesteps
        noise_schedule: Noise schedule (array of β_t values)
        embed_dim: Dimension of the embedding vector
    Returns:
        Loss for the diffusion process
    """
    noisy_image, noise, t = forward_diffusion(x_0, timesteps, noise_schedule)
    
    # Embed the timestep for conditioning
    embed = torch.nn.functional.one_hot(t, num_classes=timesteps).float()  # (batch_size, timesteps)
    embed = embed.to(x_0.device)  # Ensure it’s on the same device as the model
    
    # Predict noise using the model
    predicted_noise = model(noisy_image, embed)
    
    # Compute the mean squared error (MSE) between predicted and true noise
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
    return loss



#training
def train_diffusion_model(model, dataloader, timesteps, embed_dim, epochs=10, lr=1e-4, save_path="unet_model.pth"):
    """
    Args:
        model: U-Net model
        dataloader: PyTorch DataLoader for training data
        timesteps: Number of timesteps
        embed_dim: Dimension of the embedding vector
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save the trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise_schedule_values = noise_schedule(timesteps).to(next(model.parameters()).device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            x_0 = batch[0].to(next(model.parameters()).device)  # Input images
            
            # Compute loss
            loss = diffusion_loss(model, x_0, timesteps, noise_schedule_values, embed_dim)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def reverse_diffusion(model, timesteps, embed_dim, img_shape):
    """
    Args:
        model: Trained U-Net model
        timesteps: Number of timesteps
        embed_dim: Dimension of the embedding vector
        img_shape: Shape of the output image (C, H, W)
    Returns:
        Generated image
    """
    noise_schedule_values = noise_schedule(timesteps).to(next(model.parameters()).device)
    alpha = 1.0 - noise_schedule_values
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # Start with random noise
    x_t = torch.randn((1, *img_shape)).to(next(model.parameters()).device)

    for t in range(timesteps - 1, -1, -1):
        t_tensor = torch.tensor([t]).to(x_t.device)
        
        # Embed the timestep
        embed = torch.nn.functional.one_hot(t_tensor, num_classes=timesteps).float().to(x_t.device)
        embed = torch.nn.Linear(timesteps, embed_dim).to(x_t.device)(embed)
        
        # Predict noise
        predicted_noise = model(x_t, embed)
        
        # Compute denoised image
        alpha_bar_t = alpha_bar[t]
        alpha_t = alpha[t]
        beta_t = noise_schedule_values[t]
        
        x_t = (
            1 / torch.sqrt(alpha_t)
            * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)
        )
        
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = x_t + torch.sqrt(beta_t) * noise
    
    return x_t

# if __name__ == "__main__":
#     import torchvision.transforms as transforms
#     from torchvision.datasets import CIFAR10
#     from torch.utils.data import DataLoader

#     # Parameters
#     input_channels = 3
#     output_channels = 3
#     embed_dim = 128
#     timesteps = 1000
#     epochs = 5  # You can adjust this for quicker training
#     batch_size = 32
#     lr = 1e-4

#     # Dataset and DataLoader (example with CIFAR-10)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((64, 64))  # Resize to match U-Net input
#     ])
#     train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     # Initialize the U-Net model
#     model = UNet(input_channels, output_channels, embed_dim, timesteps).to("cuda" if torch.cuda.is_available() else "cpu")

#     # Train and save the model
#     train_diffusion_model(model, train_loader, timesteps, embed_dim, epochs=epochs, lr=lr, save_path="unet_model.pth")

if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader

    # Parameters
    input_channels = 3
    output_channels = 3
    embed_dim = 64  # Reduced from 128
    timesteps = 100  # Reduced from 1000
    epochs = 1  # Reduced for testing
    batch_size = 8  # Reduced from 32
    lr = 1e-4

    # Dataset and DataLoader (example with CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))  # Use smaller size for quicker processing
    ])
    train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define a lightweight U-Net model
    class LightweightUNet(UNet):
        def __init__(self, input_channels, output_channels, embed_dim, timesteps):
            super().__init__(input_channels, output_channels, embed_dim, timesteps)
            # Modify channel counts for lighter architecture
            self.down1 = DownsampleBlock(32, 64, embed_dim)
            self.down2 = DownsampleBlock(64, 128, embed_dim)
            self.down3 = DownsampleBlock(128, 256, embed_dim)
            self.up3 = UpsampleBlock(256, 128, embed_dim, skip_channels=128)
            self.up2 = UpsampleBlock(128, 64, embed_dim, skip_channels=64)
            self.up1 = UpsampleBlock(64, 32, embed_dim, skip_channels=32)
            self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)

    # Initialize the lightweight U-Net model
    model = LightweightUNet(input_channels, output_channels, embed_dim, timesteps).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Train and save the model
    train_diffusion_model(model, train_loader, timesteps, embed_dim, epochs=epochs, lr=lr, save_path="lightweight_unet_model.pth")