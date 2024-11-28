from dataset import load_data
import logging
import torch
from tqdm import tqdm
from models import ModelHandler
from utils import plot_real_images, generate_images, fetch_model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    model: ModelHandler,
    epochs: int,
    data_loader: torch.utils.data.DataLoader
):  
    n_batches = len(data_loader)
    train_losses = []
    gen_losses, critic_losses, gp_losses = [], [], []
    
    # train loop
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = 0
        batch_iterator = tqdm(
            enumerate(data_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            total=n_batches,
            leave=False
        )
    
        for _ , (data, target) in batch_iterator:
            # convert to device
            data, target = data.to(device), target.to(device)
            # forward pass
            critic_loss, gen_loss, \
                gradient_penalty, gradients = model(data)
            # update losses
            # append if not none
            if critic_loss is not None:
                critic_losses.append(critic_loss)
            if gen_loss is not None:
                gen_losses.append(gen_loss)
            if gradient_penalty is not None:
                gp_losses.append(gradient_penalty)
            train_loss += (critic_loss + gen_loss + gradient_penalty)
            
            # Update batch progress bar
            batch_iterator.set_postfix({
                'gen_loss': f'{sum(gen_losses)/len(gen_losses):.4f}' if gen_losses else 'N/A',
                'critic_loss': f'{sum(critic_losses)/len(critic_losses):.4f}' if critic_losses else 'N/A',
                'gradient_penalty': f'{sum(gp_losses)/len(gp_losses):.4f}' if gp_losses else 'N/A'
            })
            
        train_losses.append(train_loss/n_batches)

        # generate images
        with torch.no_grad():
            generate_images(wgan.gen, device, epoch, z_dim)
        
        # Update epoch progress bar with running means
        tqdm.write(
            f"Epoch {epoch+1}/{epochs} | "
            f"Gen Loss: {sum(gen_losses)/len(gen_losses):.4f} | "
            f"Critic Loss: {sum(critic_losses)/len(critic_losses):.4f} | "
            f"Gradient Penalty: {sum(gp_losses)/len(gp_losses):.4f} | "
            f"Train Loss: {train_losses[-1]:.4f}"
        )

    
if __name__ == "__main__":
    logger.info("Loading data")
    
    # Hyperparameters
    batch_size = 128
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    z_dim = 64
    lambda_gp = 10
    visualize = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, \
        test_loader = load_data(batch_size=batch_size)
    
    if visualize:
        logger.info("Plotting real images")
        # plot real images
        for data, target in train_loader:
            print("Data shape: ", data.shape, "Min: ", data.min(), "Max: ", data.max())
            plot_real_images(data, target)
            break

    logger.info("Initializing model")
    model = ModelHandler(model_name="vae")
    
    
    logger.info("Training model")
    train(
        model=model,
        epochs=50,
        data_loader=train_loader
    )
   
