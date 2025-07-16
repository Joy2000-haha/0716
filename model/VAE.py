import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super(VAE, self).__init__()

        # Encoder layers
        # Input: 210x160x3
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        # Output: 105x80x32

        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # Output: 52x40x64

        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        # Output: 26x20x128

        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        # Output: 13x10x256

        # Calculate flattened size
        self.flatten_size = 13 * 10 * 256  # 33,280

        # Latent space layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder layers
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        # Decoder convolutional layers
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        # Output: 26x20x128

        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        # Output: 52x40x64

        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2, output_padding=0)
        # Output: 104x80x32

        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=1, output_padding=0)
        # Output: 210x160x3

    def encode(self, x):
        # Encoder forward pass
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))

        # Flatten
        h = h.view(h.size(0), -1)

        # Get mu and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + sigma * epsilon
        # epsilon ~ N(0, 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # Decoder forward pass
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 13, 10)

        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))

        # Final layer with sigmoid activation
        reconstruction = torch.sigmoid(self.dec_conv4(h))

        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


# Loss function for VAE
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence

    Args:
        recon_x: reconstructed images
        x: original images
        mu: latent mean
        logvar: latent log variance
        beta: weight for KL divergence term (beta-VAE)
    """
    # Reconstruction loss (binary cross-entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) ~ N(0, 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD


# Example usage
if __name__ == "__main__":
    # Create model
    model = VAE(latent_dim=1024)

    # Example input
    batch_size = 16
    example_input = torch.randn(batch_size, 3, 210, 160)

    # Forward pass
    reconstruction, mu, logvar = model(example_input)

    print(f"Input shape: {example_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Calculate loss
    loss = vae_loss(reconstruction, example_input, mu, logvar)
    print(f"VAE Loss: {loss.item()}")

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")