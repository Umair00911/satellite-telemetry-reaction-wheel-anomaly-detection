import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class TelemetryGenerator(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super(TelemetryGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, feature_dim)
        )

    def forward(self, z):
        return self.model(z)

class TelemetryDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(TelemetryDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feature_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

class GANAugmentor:
    def __init__(self, feature_dim, latent_dim=64):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = TelemetryGenerator(latent_dim, feature_dim).to(self.device)
        self.discriminator = TelemetryDiscriminator(feature_dim).to(self.device)
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def gradient_penalty(self, real, fake):
        batch = real.size(0)
        alpha = torch.rand(batch, 1, device=self.device).expand_as(real)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        
        scores = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def train_gan(self, X_anomaly, epochs=70, batch_size=64, lr=0.0002):
        """Trains WGAN-GP on anomaly samples with n_critic updates."""
        print("\n🎨 Training WGAN-GP Generator for Synthetic Anomaly Augmentation...")
        if len(X_anomaly) == 0:
            print("⚠️ No anomaly data passed to GAN.")
            return

        dataset = torch.utils.data.TensorDataset(torch.tensor(X_anomaly, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.0, 0.9))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

        best_loss = float("inf")

        for epoch in range(epochs):
            for (real_x,) in dataloader:
                real_x = real_x.to(self.device)
                b_size = real_x.size(0)

                # Train Discriminator (Critic) - 5 updates
                for _ in range(5):
                    opt_d.zero_grad()
                    z = torch.randn(b_size, self.latent_dim, device=self.device)
                    fake_x = self.generator(z)

                    d_real = self.discriminator(real_x).mean()
                    d_fake = self.discriminator(fake_x.detach()).mean()
                    
                    gp = self.gradient_penalty(real_x, fake_x.detach())
                    loss_d = d_fake - d_real + 10 * gp
                    
                    loss_d.backward()
                    opt_d.step()

                # Train Generator - 1 update
                opt_g.zero_grad()
                z = torch.randn(b_size, self.latent_dim, device=self.device)
                fake_x = self.generator(z)
                d_fake_g = self.discriminator(fake_x).mean()
                loss_g = -d_fake_g
                loss_g.backward()
                opt_g.step()

            scheduler_g.step()
            scheduler_d.step()

            if loss_g.item() < best_loss:
                best_loss = loss_g.item()
                torch.save(self.generator.state_dict(), "best_generator.pt")

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"  GAN Epoch [{epoch+1}/{epochs}] - Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

        print("✅ WGAN-GP Training Complete. Best generator saved.")

    def generate_synthetic_anomalies(self, num_samples):
        """Synthesizes num_samples fake anomaly telemetry vectors."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            synth_x = self.generator(z).cpu().numpy()
        return synth_x

def augment_dataset(X_train, y_train, gan_augmentor, num_synth=100):
    """Augments training dataset with GAN-generated synthetic anomalies."""
    print(f"✨ Generating {num_synth} synthetic anomaly samples via WGAN-GP...")
    synth_anomalies = gan_augmentor.generate_synthetic_anomalies(num_synth)
    synth_labels = np.ones(num_synth, dtype=int)

    X_augmented = np.vstack([X_train, synth_anomalies])
    y_augmented = np.hstack([y_train, synth_labels])

    perm = np.random.permutation(len(y_augmented))
    return X_augmented[perm], y_augmented[perm]