# model selection
from models.gan import WassersteinGenerativeAdversarialNetwork
from models.vae import VariationalAutoencoder

# class that handles all models
class ModelHandler:
    def __init__(self, model_name: str, **kwargs):
        self.models = {
            "vae": VariationalAutoencoder(),
            "gan": WassersteinGenerativeAdversarialNetwork()
        }
        self.model_name = model_name
        if self.model_name not in self.models:
            raise ValueError(f"Model {self.model_name} not found")
        self.model = self.models[self.model_name](**kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# WassersteinGenerativeAdversarialNetwork(
#     z_dim=kwargs["z_dim"],
#     ch_out=1,
#     batch_size=kwargs["batch_size"],
#     device=kwargs["device"],
#     lambda_gp=kwargs["lambda_gp"],
#     critic_update_freq=5,
#     learning_rate=kwargs["lr"],
#     beta1=kwargs["beta_1"],
#     beta2=kwargs["beta_2"]
# )