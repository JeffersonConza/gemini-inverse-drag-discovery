# %% [markdown]
# # Inverse Drag Discovery: Physics-Informed Neural Networks (PINNs)
# **Proyecto:** Descubrimiento de Masa y Coeficiente de Arrastre mediante SciML.
# **Autor:** SciML Research Lead (Gemini CLI)
#
# Este cuaderno unifica la simulación física, la generación de telemetría ruidosa 
# y el entrenamiento de una PINN para resolver un problema inverso multiparamétrico.

# %% [1] Configuración e Importaciones
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import io
import csv
from typing import Callable, Any

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo activo: {device}")

# %% [2] Motor de Física e Integración Numérica
# Definimos el integrador RK4 y las ecuaciones diferenciales del cohete.

def rk4_step(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt/2 * k1, *args)
    k3 = f(t + dt/2, y + dt/2 * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def cinematica_cohete(t, state, masa, empuje, angulo_deg, cd):
    x, y_alt, vx, vy = state
    G, RHO_0, H, AREA = 9.81, 1.225, 8500.0, 3.14
    
    # Gravity Turn
    theta = np.radians(angulo_deg) if t < 20.0 else np.radians(max(0.0, angulo_deg - (t-20.0)*2.0))
    
    # Arrastre
    rho = RHO_0 * np.exp(-max(0.0, y_alt) / H) if y_alt < 100000.0 else 0.0
    v_mag = np.sqrt(vx**2 + vy**2) + 1e-8
    f_drag = 0.5 * rho * v_mag**2 * cd * AREA
    
    # Derivadas
    dvx = (empuje * np.cos(theta) - f_drag * (vx / v_mag)) / masa
    dvy = (empuje * np.sin(theta) - f_drag * (vy / v_mag)) / masa - G
    return np.array([vx, vy, dvx, dvy])

# %% [3] Generación de Telemetría Sintética (Data Engineering)
# Simulamos un vuelo real y degradamos las señales con ruido de sensores.

print("🧪 Generando telemetría con ruido instrumental...")
MASA_REAL, CD_REAL, ANGULO_INI = 500.0, 0.3, 85.0
t, dt, t_final = 0.0, 0.1, 150.0
state = np.array([0.0, 0.0, 0.0, 0.0])
datos_limpios = []

while t <= t_final and state[1] >= 0:
    datos_limpios.append([t, state[0], state[1], state[2], state[3]])
    state = rk4_step(cinematica_cohete, t, state, dt, MASA_REAL, 15000.0, ANGULO_INI, CD_REAL)
    t += dt

# Adición de ruido gaussiano (σ_y=2.0m, σ_v=0.1m/s)
datos = np.array(datos_limpios)
datos[:, 2] += np.random.normal(0, 2.0, size=len(datos)) # Altitud
datos[:, 3:5] += np.random.normal(0, 0.1, size=(len(datos), 2)) # Velocidades

# Guardar y previsualizar
df_vuelo = pd.DataFrame(datos, columns=['t', 'x', 'y', 'vx', 'vy'])
df_vuelo.to_csv('vuelo_colab.csv', index=False)
plt.scatter(df_vuelo['t'], df_vuelo['y'], s=1, color='gray', label='Telemetría Ruidosa')
plt.title("Datos de Entrada para la PINN")
plt.legend()
plt.show()

# %% [4] Arquitectura de la PINN y Física Informada
# Definimos la red neuronal y el funcional de pérdida física (EDO Loss).

class PINN_Drag(nn.Module):
    def __init__(self):
        super(PINN_Drag, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)
        )
        # Parámetros a descubrir: Masa y Coeficiente de Arrastre
        self.cd = nn.Parameter(torch.tensor([0.05]))   # Suposición inicial: 0.05
        self.masa = nn.Parameter(torch.tensor([300.0])) # Suposición inicial: 300kg

    def forward(self, t):
        return self.net(t)

def physical_loss(t, state_pred, cd_pred, masa_pred):
    x, y, vx, vy = state_pred[:, 0:1], state_pred[:, 1:2], state_pred[:, 2:3], state_pred[:, 3:4]
    
    # Diferenciación automática respecto al tiempo
    def grad(u): return torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dx_dt, dy_dt, dvx_dt, dvy_dt = grad(x), grad(y), grad(vx), grad(vy)

    # Constantes y EDOs
    g, rho_0, H, AREA, EMPUJE = 9.81, 1.225, 8500.0, 3.14, 15000.0
    angulo_rad = np.radians(85.0)
    theta = torch.where(t < 20.0, torch.full_like(t, angulo_rad), torch.clamp(angulo_rad - (t - 20.0) * np.radians(2.0), min=0.0))
    
    v_norm = torch.sqrt(vx**2 + vy**2) + 1e-8
    rho = torch.where(y < 100000.0, rho_0 * torch.exp(-y / H), torch.zeros_like(y))
    f_d = 0.5 * rho * v_norm**2 * cd_pred * AREA
    
    # Residuos de Newton
    res_x = dx_dt - vx
    res_y = dy_dt - vy
    res_vx = dvx_dt - (EMPUJE * torch.cos(theta) - f_d * (vx / v_norm)) / masa_pred
    res_vy = dvy_dt - (EMPUJE * torch.sin(theta) - f_d * (vy / v_norm)) / masa_pred + g
    
    return torch.mean(res_x**2 + res_y**2 + res_vx**2 + res_vy**2)

# %% [5] Entrenamiento y Optimización Multiparamétrica
# El modelo aprende la trayectoria y descubre la física simultáneamente.

t_tensor = torch.tensor(df_vuelo['t'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)
s_tensor = torch.tensor(df_vuelo[['x', 'y', 'vx', 'vy']].values, dtype=torch.float32)
s_mean, s_std = s_tensor.mean(dim=0), s_tensor.std(dim=0) + 1e-8
s_norm = (s_tensor - s_mean) / s_std

model = PINN_Drag()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
frames = []

print("\n🚀 Iniciando entrenamiento (Aprox. 2-5 mins en CPU)...")
for epoch in range(3001):
    optimizer.zero_grad()
    s_pred_norm = model(t_tensor)
    s_pred = s_pred_norm * s_std + s_mean
    
    l_data = nn.MSELoss()(s_pred_norm, s_norm)
    l_phys = physical_loss(t_tensor, s_pred, model.cd, model.masa)
    loss = l_data + (1e-5 * l_phys)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"E{epoch:04d} | Loss: {loss.item():.5f} | Cd: {model.cd.item():.3f} | Masa: {model.masa.item():.1f}kg")
        # Captura de frame para animación
        with torch.no_grad():
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(t_tensor.detach(), s_tensor[:, 1], 'k', alpha=0.2)
            plt.plot(t_tensor.detach(), s_pred[:, 1], 'b')
            plt.title(f"Altitud Epoch {epoch}")
            plt.subplot(1, 2, 2)
            plt.plot(t_tensor.detach(), s_tensor[:, 3], 'k', alpha=0.2)
            plt.plot(t_tensor.detach(), s_pred[:, 3], 'r')
            plt.title(f"Cd={model.cd.item():.3f} m={model.masa.item():.1f}")
            buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
            frames.append(imageio.v2.imread(buf)); plt.close()

# %% [6] Resultados y Exportación
print(f"\n✅ Valor Real: Cd=0.30 | Masa=500kg")
print(f"🤖 Descubierto: Cd={model.cd.item():.4f} | Masa={model.masa.item():.2f}kg")

imageio.mimsave('convergencia_colab.gif', frames, fps=10)
print("🎥 Animación guardada como 'convergencia_colab.gif'")
