import torchvision
import matplotlib.pyplot as plt
import torch

def generate_images(generator, device, epoch, z_dim: int):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, z_dim).to(device)
        images = generator(noise)
        # -1 and 1 to 0 and 1
        images = (images + 1) / 2
    # Convert images back to CPU and create grid
    images = images.cpu()
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    
    # Convert to matplotlib-compatible format
    grid = grid.permute(1, 2, 0)
    
    # Plot and save
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f'output/generated_images_epoch_0.png')
    plt.close()

def plot_real_images(data, target):
    images = data.cpu()[:16]
    # -1 and 1 to 0 and 1
    images = (images + 1) / 2
    target = target.cpu()[:16]
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis('off')
    # Add title with labels
    labels = target.numpy()
    title = f'Real Images (Labels: {labels})'
    plt.title(title)
    plt.savefig(f'output/real_images.png')
    plt.close()