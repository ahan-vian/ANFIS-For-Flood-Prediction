import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. Setup ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- 2. Load Dataset ---
df = pd.read_csv("dummy_flood_data4.csv")
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = df[['rainfall', 'water_level']].values
y = df['flood_potential'].values.reshape(-1, 1)
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y).flatten()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=SEED)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# --- 3. Membership Function ---
class TrainableGaussianMF(nn.Module):
    def __init__(self, num_mf):
        super().__init__()
        self.mean = nn.Parameter(torch.linspace(0.2, 0.8, num_mf))
        self._raw_sigma = nn.Parameter(torch.full((num_mf,), -1.0))  # raw log-sigma

    def forward(self, x):
        sigma = torch.nn.functional.softplus(self._raw_sigma) + 1e-3  # jamin >0
        x = x.unsqueeze(1)
        return torch.exp(-0.5 * ((x - self.mean) / sigma) ** 2)

# --- 4. ANFIS Model ---
class ANFISNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_mf = 3
        self.num_rules = self.num_mf ** 2
        self.mf_rain = TrainableGaussianMF(self.num_mf)
        self.mf_level = TrainableGaussianMF(self.num_mf)
        self.a1 = nn.Parameter(torch.randn(self.num_rules))
        self.a2 = nn.Parameter(torch.randn(self.num_rules))
        self.b = nn.Parameter(torch.randn(self.num_rules))

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        mf1 = self.mf_rain(x1)
        mf2 = self.mf_level(x2)
        rule_activation = torch.einsum("bi,bj->bij", mf1, mf2).reshape(-1, self.num_rules)
        weights = rule_activation / rule_activation.sum(dim=1, keepdim=True)
        f = (self.a1 * x1.unsqueeze(1)) + (self.a2 * x2.unsqueeze(1)) + self.b
        f = torch.tanh(f)
        y = torch.sum(weights * f, dim=1)
        return y

# --- 5. PSO Training ---
model = ANFISNet()
param_shapes = [p.shape for p in model.parameters()]
param_sizes = [p.numel() for p in model.parameters()]
total_params = sum(param_sizes)

def set_model_params(model, vector):
    idx = 0
    with torch.no_grad():
        for p, shape, size in zip(model.parameters(), param_shapes, param_sizes):
            val = torch.tensor(vector[idx:idx+size], dtype=torch.float32).reshape(shape)
            p.copy_(val)
            idx += size

def evaluate_fitness(vector):
    set_model_params(model, vector)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train_tensor).squeeze()
        loss = mean_squared_error(y_train, y_pred.numpy())
    return loss

num_particles = 30
num_iterations = 100
w = 0.7
c1 = c2 = 1.5

positions = np.random.uniform(-1, 1, (num_particles, total_params))
velocities = np.zeros_like(positions)
p_best = positions.copy()
p_best_scores = np.array([evaluate_fitness(p) for p in positions])
g_best = p_best[np.argmin(p_best_scores)]
g_best_score = np.min(p_best_scores)

pso_losses = []
for it in range(num_iterations):
    for i in range(num_particles):
        fitness = evaluate_fitness(positions[i])
        if fitness < p_best_scores[i]:
            p_best_scores[i] = fitness
            p_best[i] = positions[i].copy()
            if fitness < g_best_score:
                g_best = positions[i].copy()
                g_best_score = fitness
    r1 = np.random.rand(*positions.shape)
    r2 = np.random.rand(*positions.shape)
    velocities = w * velocities + c1 * r1 * (p_best - positions) + c2 * r2 * (g_best - positions)
    positions += velocities
    pso_losses.append(g_best_score)
    if it % 10 == 0:
        print(f"Iter {it:3d} | Best RMSE: {np.sqrt(g_best_score):.6f}")

set_model_params(model, g_best)

# --- 6. Evaluation ---
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).squeeze().numpy()
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

# --- 7. Visualization ---
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color="green")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Flood Potential")
plt.ylabel("Predicted (ANFIS PSO)")
plt.title(f"ANFIS PSO Regression Result\nRMSE: {rmse:.4f}, R²: {r2:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(pso_losses, label='PSO Best Loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("PSO Optimization Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test, 'r--', label='Original Data')
plt.plot(y_pred_test, 'b-', label='PSO Predicted')
plt.xlabel('Number of Test Samples')
plt.ylabel('Flood Potential')
plt.title('ANFIS PSO Output vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
