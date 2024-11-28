# load mnist data
import torch
import torchvision
import torchvision.transforms as transforms

# Define transformations to convert to tensor and rescale to [-1,1]
transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1] using mean=0.5, std=0.5
])

def load_data(batch_size: int):
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_to_tensor)
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_to_tensor)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True), torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)


if __name__ == '__main__':
    # print batch shape
    train_loader, test_loader = load_data(batch_size=64)
    for batch in train_loader:
        print(batch[0].shape)
        break
