import torch
import torch.nn as nn

# Define same generator architecture as in training
class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x).view(-1, 1, 28, 28)

# Load model function
def load_generator(weights_path="generator.pth"):
    model = Generator()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# One-hot utility
def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]
