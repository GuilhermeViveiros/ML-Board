import torch
from typing import List
from models.wasserstein import calculate_gradient_penalty

class Critic(torch.nn.Module):
    def __init__(self, ch_in: int):
        super(Critic, self).__init__()
        self.cv1 = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(ch_in, 64, kernel_size=4, stride=2)
        )
        self.cv2 = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
        )
        self.cv3 = torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(128, 1, kernel_size=4, stride=2)
        )
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.cv1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        x = self.cv2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        x = self.cv3(x)
    
        return x.view(len(x), -1)

class Discriminator(torch.nn.Module):
    def __init__(self, ch_in: int):
        super(Discriminator, self).__init__()
        self.cv1 = torch.nn.Conv2d(ch_in, 64, kernel_size=4, stride=2, padding=1)
        self.cv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.cv3 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.flatten = torch.nn.Flatten()
        
        self.fc1 = torch.nn.Linear(2304, 1)
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.cv1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        x = self.cv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        x = self.cv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        #import pdb; pdb.set_trace()
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        #import pdb; pdb.set_trace()
        return torch.sigmoid(x)

class Generator(torch.nn.Module):
    def __init__(self, z_dim: int = 100):

        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        self.cv1 = torch.nn.ConvTranspose2d(z_dim, 128, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.cv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.cv3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.cv4 = torch.nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU(inplace=True)
        # expand one dim

            
    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        x = self.cv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.cv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.cv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.cv4(x)
        return torch.tanh(x)

class WassersteinGenerativeAdversarialNetwork():
    def __init__(
        self,
        z_dim: int,
        ch_out: int,
        batch_size: int,
        device: str,
        lambda_gp: float = 10.0,
        critic_update_freq: int = 5,
        learning_rate: float = 0.00005,  # Reduced learning rate
        beta1: float = 0.0,  # Changed beta1 to 0.0 as per WGAN-GP paper
        beta2: float = 0.9   # Changed beta2 to 0.9
    ):
        self.z_dim = z_dim
        #self.disc = Discriminator(ch_in=ch_out).to(device)
        #self.disc.train()
        self.critic = Critic(ch_in=ch_out)
        self.gen = Generator(z_dim=z_dim)
        self.critic.apply(self.weights_init)
        self.gen.apply(self.weights_init)
        self.gen.to(device)
        self.critic.to(device)
        # parameters
        self.batch_size = batch_size
        self.device = device
        self.lambda_gp = lambda_gp
        self.critic_update_freq = critic_update_freq
        # Use much smaller learning rates
        self.critic_lr = learning_rate  # Reduced significantly
        self.gen_lr = learning_rate    # Reduced significantly
        self.beta1 = beta1
        self.beta2 = beta2
        self.__optimizers__()

    def __optimizers__(self):
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            betas=(self.beta1, self.beta2)
        )
        self.generator_optimizer = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.gen_lr,
            betas=(self.beta1, self.beta2)
        )
        # self.gen_loss = torch.nn.BCELoss()

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    # def train_discrimator(
    #     self,
    #     data,
    #     noise,
    # ):  
    #     # if discriminator does not exists    
    #     if self.disc is None:
    #         raise ValueError("Discriminator does not exist")
    #     # Train the discriminator
    #     self.discrimantor.zero_grad()
    #     # get generator output
    #     x_sample = self.gen(noise).detach() # avoid training the generator here
    #     # get discriminator output
    #     disc_real = self.disc(data)
    #     disc_real_labels = torch.ones_like(disc_real)
    #     disc_fake = self.disc(x_sample)
    #     disc_fake_labels = torch.zeros_like(disc_fake)

    #     # add them together and shuffle
    #     disc_loss_inputs = torch.cat([disc_real, disc_fake])
    #     disc_loss_labels = torch.cat([disc_real_labels, disc_fake_labels])
    #     # shuffle
    #     indices = torch.randperm(disc_loss_inputs.size(0))
    #     disc_loss_inputs = disc_loss_inputs[indices]
    #     disc_loss_labels = disc_loss_labels[indices]

    #     # compute discriminator loss
    #     disc_loss = criterion(disc_loss_inputs, disc_loss_labels)
    #     # backpropagate & update weights
    #     disc_loss.backward()
    #     disc_optimizer.step()
    #     return disc_loss.item()

    def train_critic(self, data, noise):
        # Track the last critic loss and penalties for return value
        last_critic_loss = 0
        last_gradient_penalty = 0
        last_gradients = None

        for _ in range(self.critic_update_freq):
            self.critic_optimizer.zero_grad()
            
            # Generate new noise for each iteration
            # curr_noise = torch.randn(self.batch_size, self.z_dim).to(self.device)
            
            # Generate fake samples with new noise
            x_fake = self.gen(noise).detach()
            
            # Get critic predictions
            critic_real = self.critic(data)
            critic_fake = self.critic(x_fake)
            
            # Compute Wasserstein loss
            critic_loss = torch.mean(critic_fake) - torch.mean(critic_real)
            
            # Compute gradient penalty
            gradient_penalty, gradients = calculate_gradient_penalty(
                self.critic, data, x_fake, self.device
            )
            
            # Total loss
            total_loss = critic_loss + self.lambda_gp * gradient_penalty
            
            # Backprop and optimize
            total_loss.backward()
            self.critic_optimizer.step()

            # Store the last iteration's values
            last_critic_loss = critic_loss.item()
            last_gradient_penalty = gradient_penalty.item()
            last_gradients = gradients

        return last_critic_loss, last_gradient_penalty, last_gradients

    def train_generator(self, noise):
        self.generator_optimizer.zero_grad()
        
        # Generate fake samples
        fake_data = self.gen(noise)
        
        # Get critic predictions
        critic_fake = self.critic(fake_data)
        
        # Generator loss (note the negative sign)
        gen_loss = -torch.mean(critic_fake)
        
        # Backprop and optimize
        gen_loss.backward()
        self.generator_optimizer.step()
        
        return gen_loss.item()

    def __call__(self, x: torch.Tensor):
        # Create noise
        noise = torch.randn(self.batch_size, self.z_dim).to(self.device)
        
        # Train critic first
        critic_loss, gradient_penalty, gradients = self.train_critic(x, noise)
        
        # Then train generator
        gen_loss = self.train_generator(noise)
        
        return critic_loss, gen_loss, gradient_penalty, gradients

if __name__ == "__main__":
    gen = Generator(z_dim=100)
    dummy_input = torch.randn(8, 100)
    outputs = gen(dummy_input)
    print("Generator output shape: ", outputs.shape) 
    disc = Critic(ch_in=1)
    outputs = disc(outputs)
    print("Discriminator output shape: ", outputs.shape)
