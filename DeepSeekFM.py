import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# Synthetic Turbulence Dataset (Example)
# ===========================================
class TurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, grid_size=64):
        self.num_samples = num_samples
        self.grid_size = grid_size
        
        # Generate synthetic turbulence data (replace with real DNS/LES data)
        # Example: Gaussian random fields with energy spectrum ~k^(-5/3)
        self.data = []
        for _ in range(num_samples):
            k = np.fft.fftfreq(grid_size) * grid_size
            kx, ky = np.meshgrid(k, k)
            energy_spectrum = (kx**2 + ky**2 + 1e-6)**(-5/6)  # Kolmogorov -5/3 spectrum
            phase = np.random.rand(grid_size, grid_size) * 2 * np.pi
            field = np.fft.ifft2(np.sqrt(energy_spectrum) * np.exp(1j * phase)).real
            self.data.append(field)
        
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32).unsqueeze(1)  # [B, 1, H, W]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# ===========================================
# Flow Matching Model (Simplified UNet)
# ===========================================
class FlowMatchingModel(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # For simplicity, we ignore time conditioning here (add embeddings in real applications)
        return self.net(x)

# ===========================================
# Training Loop
# ===========================================
def train_flow_matching(model, dataset, epochs=100, batch_size=32):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            # Sample from prior distribution (e.g., Gaussian noise)
            prior = torch.randn_like(batch)
            
            # Random time steps t ∈ [0, 1]
            t = torch.rand(batch.size(0), 1, 1, 1).to(batch.device)
            
            # Interpolate between prior and data: x_t = (1 - t) * prior + t * data
            x_t = (1 - t) * prior + t * batch
            
            # Predict the vector field (velocity)
            pred_v = model(x_t, t)
            
            # Flow Matching loss: ||pred_v - (data - prior)||^2
            target_v = batch - prior
            loss = loss_fn(pred_v, target_v)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ===========================================
# Generate Turbulence Samples
# ===========================================
def generate_turbulence(model, prior, num_steps=50):
    x = prior
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.tensor([step / num_steps], dtype=torch.float32).view(1, 1, 1, 1)
        v = model(x, t)
        x = x + v * dt
    return x

# ===========================================
# Main Workflow
# ===========================================
if __name__ == "__main__":
    # Initialize dataset and model
    dataset = TurbulenceDataset(num_samples=1000, grid_size=64)
    model = FlowMatchingModel()
    
    # Train the model
    train_flow_matching(model, dataset, epochs=50, batch_size=32)
    
    # Generate a sample
    prior = torch.randn(1, 1, 64, 64)  # Initial noise
    generated_turbulence = generate_turbulence(model, prior)
    
    # Visualize
    plt.imshow(generated_turbulence.squeeze().detach().numpy(), cmap='viridis')
    plt.title("Generated Turbulence")
    plt.colorbar()
    plt.show()