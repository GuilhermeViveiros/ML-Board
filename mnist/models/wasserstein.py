import torch

# lets make a function to calculate the gradients of the critic
def calculate_gradients(critic, real_samples, fake_samples, device):
    epsilon = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (epsilon * real_samples + ((1 - epsilon) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates), 
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return gradients
    
# calculate gradient penaly
def gradient_penalty(gradients):
    # Flatten the gradients so that each row captures one image
    gradients = gradients.view(len(gradients), -1)
    return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    

def calculate_gradient_penalty(critic, real_samples, fake_samples, device, return_gradients=True):
    gradients = calculate_gradients(critic, real_samples, fake_samples, device)
    gp = gradient_penalty(gradients)
    if return_gradients:
        return gp, gradients
    return gp


if __name__ == "__main__":
    from models.gan import Critic
    critic = Critic(ch_in=1)
    real_samples = torch.randn(1, 1, 28, 28)
    fake_samples = torch.randn(1, 1, 28, 28)
    gradients = calculate_gradients(critic, real_samples, fake_samples, "cpu")
    print("Gradients mean: ", gradients.mean())
    print("Gradients std: ", gradients.std())
    print("Gradients norm: ", gradients.norm(2, dim=1).mean())
    print("Gradients shape: ", gradients.shape)
    gp = gradient_penalty(gradients)
    print("Gradient penalty: ", gp)

