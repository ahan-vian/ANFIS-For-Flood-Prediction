# --- 1. Setup and Seed for Reproducibility ---
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 2. Import Additional Library ---
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 3. Load Dataset ---
df = pd.read_csv("dummy_flood_data.csv")
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = df[['rainfall', 'water_level']].values
y = df['flood_potential'].values.reshape(-1, 1)
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y).flatten()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=SEED)

# --- 4. Definition of Membership Function (Gaussian MF) ---
class TrainableGaussianMF(nn.Module):
    def __init__(self, num_mf):
        super().__init__()
        self.mean = nn.Parameter(torch.linspace(0.2, 0.8, num_mf))  # shape: [num_mf]
        self.sigma = nn.Parameter(torch.full((num_mf,), 0.1))       # shape: [num_mf]

    def forward(self, x):  # x shape: [batch_size]
        x = x.unsqueeze(1)  # shape: [batch_size, 1]
        return torch.exp(-0.5 * ((x - self.mean) / self.sigma)**2)  # [batch_size, num_mf]

# --- 5. ANFIS Classic Model (Sugeno-type) ---
class ANFISNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_mf = 3
        self.num_rules = self.num_mf ** 2

        # The membership function for rainfall and water level
        self.mf_rain = TrainableGaussianMF(self.num_mf)
        self.mf_level = TrainableGaussianMF(self.num_mf)

        # Consequence function parameter
        self.a1 = nn.Parameter(torch.randn(self.num_rules))
        self.a2 = nn.Parameter(torch.randn(self.num_rules))
        self.b = nn.Parameter(torch.randn(self.num_rules))

    def forward(self, x):  # x shape: [batch_size, 2]
        x1, x2 = x[:, 0], x[:, 1]
        mf1 = self.mf_rain(x1)     # shape: [batch_size, 3]
        mf2 = self.mf_level(x2)    # shape: [batch_size, 3]

        # Fuzzy rule combination: outer product per sample
        rule_activation = torch.einsum("bi,bj->bij", mf1, mf2).reshape(-1, self.num_rules)
        weights = rule_activation / rule_activation.sum(dim=1, keepdim=True)

        # Linear consequential function (per rule)
        f = (self.a1 * x1.unsqueeze(1)) + (self.a2 * x2.unsqueeze(1)) + self.b  # broadcasting
        f = torch.tanh(f)  # Boleh dihapus jika tidak ingin batasi output
        y = torch.sum(weights * f, dim=1)
        return y

# --- 6. ANFIS Model Training ---
model = ANFISNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

losses = []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor).squeeze()
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# --- 7. Testing and Evaluation ---
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_test = model(X_test_tensor).squeeze().numpy()

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

# --- 8. Visualization of Prediction Results ---
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color="green")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Flood Potential")
plt.ylabel("Predicted (ANFIS)")
plt.title(f"ANFIS Regression Result\nRMSE: {rmse:.4f}, R²: {r2:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 9.  Visualization Loss Training ---
plt.figure(figsize=(8,4))
plt.plot(losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANFIS Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test, 'r--', label='Original Data')
plt.plot(y_pred_test, 'b-', label='Obtained Data')
plt.xlabel('Number of Test Samples')
plt.ylabel('Flood Potential')
plt.title('ANFIS Output vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
