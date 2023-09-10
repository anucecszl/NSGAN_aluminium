import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Set device preference. CPU is more efficient for smaller networks.
gpu = False
device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

# Load and convert the mechanical properties dataset to a numpy array.
data_df = pd.read_excel('Al_mechanical_dataset.xlsx')
data_np = data_df.to_numpy()

# Normalize alloy composition features for consistent model input.
comp_data = data_np[:, 5:40].astype(float)
comp_min = np.min(comp_data, axis=0)
comp_max = np.max(comp_data, axis=0)
X = (comp_data - comp_min) / comp_max


# Create a custom Dataset for the GAN's training loop.
class GANTrainSet(Dataset):
    def __init__(self):
        self.features = torch.from_numpy(X).float()
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.len


# Define the Generator network with three fully-connected layers.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 35),
            nn.ReLU(),
            nn.Linear(35, 35),
            nn.ReLU(),
            nn.Linear(35, 35),
            nn.ReLU(),
        )

    def forward(self, noise):
        return self.model(noise)


# Define the Discriminator network.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(35, 35),
            nn.LeakyReLU(),
            nn.Linear(35, 35),
            nn.LeakyReLU(),
            nn.Linear(35, 1),
        )

    def forward(self, x):
        return self.model(x)


# Compute the gradient penalty for Wasserstein GAN with Gradient Penalty (WGAN-GP).
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == "__main__":
    lambda_gp = 0.01
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)

    al_data_set = GANTrainSet()
    loader = DataLoader(dataset=al_data_set, batch_size=5, shuffle=True)

    # Main training loop: Train the discriminator and generator iteratively.
    for epoch in range(10000):
        loss_d_real = 0
        loss_d_fake = 0
        total_d_loss = 0

        for i, alloy in enumerate(loader):
            real_input = alloy

            # Update the discriminator by iterating over real and fake samples.
            for j in range(5):
                g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
                fake_alloy = generator(g_noise)
                fake_input = fake_alloy.detach()

                optimizer_D.zero_grad()
                gradient_penalty = compute_gradient_penalty(discriminator, real_input.data, fake_input.data)
                d_loss = (-torch.mean(discriminator(real_input)) +
                          torch.mean(discriminator(fake_input)) +
                          lambda_gp * gradient_penalty)
                d_loss.backward()
                optimizer_D.step()

                loss_d_real += discriminator(real_input).sum().item()
                loss_d_fake += discriminator(fake_input).sum().item()
                total_d_loss += -d_loss.item()

            # Update the generator using the discriminator's feedback.
            g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
            fake_alloy = generator(g_noise)
            optimizer_G.zero_grad()
            g_loss = -torch.mean(discriminator(fake_alloy))
            g_loss.backward()
            optimizer_G.step()

        # Periodically display generated alloy samples for evaluation.
        if epoch % 20 == 0:
            g_noise = torch.tensor(np.random.randn(6, 10)).float()
            fake_alloy = generator(g_noise)
            print(fake_alloy.detach().numpy()[:, -10:])

        balance = loss_d_real / (loss_d_real + loss_d_fake)
        if epoch < 5000 or epoch % 20 == 0:
            print(epoch, "discriminator balance:", balance, "d_loss:", total_d_loss)
        if epoch % 999 == 0:
            g_noise = torch.tensor(np.random.randn(1000, 10)).float()
            fake_alloy = generator(g_noise)
            show_np = fake_alloy.detach().numpy()[:, -10:]
            print('mean:', np.mean(np.sum(show_np, axis=1)))
            print('std:', np.std(np.sum(show_np, axis=1)))

    # Preserve the trained generator model for future use.
    torch.save(generator.state_dict(), 'generator_net_aluminium_gp.pt')
