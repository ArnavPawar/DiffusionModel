import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from MyGnat import UNet, noise_schedule, reverse_diffusion  # Import your custom classes/functions

# Load the image
def load_image(image_path, target_size):
    img = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Converts to [C, H, W]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension [1, C, H, W]

# Save the output
def save_image(tensor, output_path):
    tensor = tensor.squeeze(0)  # Remove batch dimension [C, H, W]
    tensor = tensor * 0.5 + 0.5  # De-normalize to [0, 1]
    tensor = tensor.clamp(0, 1)  # Clip values to valid range
    img = transforms.ToPILImage()(tensor)
    img.save(output_path)

# Test with an image
def test_with_image(model_path, input_image, output_image, timesteps=1000, embed_dim=128):
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=3, output_channels=3, embed_dim=embed_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load the input image
    input_tensor = load_image(input_image, target_size=(64, 64)).to(device)

    # Define noise schedule
    noise_schedule_values = noise_schedule(timesteps).to(device)

    # Generate output using reverse diffusion
    output_tensor = reverse_diffusion(model, timesteps, embed_dim, img_shape=(3, 64, 64))

    # Save and display the results
    save_image(output_tensor, output_image)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(transforms.ToPILImage()(input_tensor.squeeze(0) * 0.5 + 0.5))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Generated Image")
    plt.imshow(Image.open(output_image))
    plt.axis("off")
    plt.show()

# Paths
input_image_path = "image.jpg"
output_image_path = "output.jpg"
model_checkpoint = "unet_model.pth"  # Ensure this path points to your trained model

# Run the test
if __name__ == "__main__":
    test_with_image(model_checkpoint, input_image_path, output_image_path)
