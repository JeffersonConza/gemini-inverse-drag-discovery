import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
from tqdm import tqdm

class PINN_Drag(nn.Module):
    def __init__(self):
        super(PINN_Drag, self).__init__()
        # Entrada: tiempo (t) -> Salida: [x, y, vx, vy]
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4) 
        )
        # 🎯 Parámetros físicos a descubrir (Problema Inverso)
        self.cd = nn.Parameter(torch.tensor([0.05]))   # Real: 0.3
        self.masa = nn.Parameter(torch.tensor([300.0])) # Real: 500.0

    def forward(self, t):
        return self.net(t)

def physical_loss(t, state_pred, cd_pred, masa_pred):
    """Residuo de las EDOs usando diferenciación automática (Autograd)"""
    x = state_pred[:, 0:1]
    y = state_pred[:, 1:2]
    vx = state_pred[:, 2:3]
    vy = state_pred[:, 3:4]
    
    # Derivadas de las predicciones de la red respecto al tiempo
    dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    dvx_dt = torch.autograd.grad(vx, t, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
    dvy_dt = torch.autograd.grad(vy, t, grad_outputs=torch.ones_like(vy), create_graph=True)[0]

    # Constantes físicas
    g = 9.81
    rho_0 = 1.225
    H = 8500.0
    area = 3.14
    empuje = 15000.0
    angulo_inicial = np.radians(85.0)
    
    theta = torch.where(
        t < 20.0, 
        torch.full_like(t, angulo_inicial), 
        torch.clamp(angulo_inicial - (t - 20.0) * np.radians(2.0), min=0.0)
    )
    
    v_norm = torch.sqrt(vx**2 + vy**2) + 1e-8
    rho = torch.where(y < 100000.0, rho_0 * torch.exp(-y / H), torch.zeros_like(y))
    F_d = 0.5 * rho * v_norm**2 * cd_pred * area
    
    # Ecuaciones residuales
    res_x = dx_dt - vx
    res_y = dy_dt - vy
    res_vx = dvx_dt - (empuje * torch.cos(theta) / masa_pred) + (F_d / masa_pred) * (vx / v_norm)
    res_vy = dvy_dt - (empuje * torch.sin(theta) / masa_pred) + g + (F_d / masa_pred) * (vy / v_norm)
    
    return torch.mean(res_x**2 + res_y**2 + res_vx**2 + res_vy**2)

if __name__ == "__main__":
    print("🚀 Iniciando Entrenamiento con Generación de Animación...")
    
    # Cargar telemetría
    df = pd.read_csv('data/vuelo.csv')
    t_data = torch.tensor(df['t'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    state_data = torch.tensor(df[['x', 'y', 'vx', 'vy']].values, dtype=torch.float32)
    
    # Normalización
    state_mean = state_data.mean(dim=0)
    state_std = state_data.std(dim=0) + 1e-8
    state_data_norm = (state_data - state_mean) / state_std
    
    model = PINN_Drag()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    frames = [] # Almacén para el GIF
    loss_history = []
    epochs = 3000
    
    pbar = tqdm(range(epochs), desc="Entrenando PINN")
    for epoch in pbar:
        optimizer.zero_grad()
        
        state_pred_norm = model(t_data)
        state_pred = state_pred_norm * state_std + state_mean
        
        loss_data = nn.MSELoss()(state_pred_norm, state_data_norm)
        loss_physics = physical_loss(t_data, state_pred, model.cd, model.masa)
        loss = loss_data + (1e-5 * loss_physics)
        loss_history.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            pbar.set_postfix({
                'Loss': f"{loss.item():.2e}",
                'Cd': f"{model.cd.item():.4f}",
                'Masa': f"{model.masa.item():.1f}kg"
            })
            
        # Capturar frame cada 100 épocas
        if epoch % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                t_num = t_data.detach().numpy()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Título dinámico profesional
                fig.suptitle(f"Entrenamiento PINN | Epoch: {epoch:04d} | Loss: {loss.item():.2e}\n"
                             f"Parámetros Descubiertos: Cd = {model.cd.item():.4f} | Masa = {model.masa.item():.1f}kg", 
                             fontsize=13, fontweight='bold', y=0.98)
                
                # Panel Izquierdo: Trayectoria (Altitud)
                ax1.plot(t_num, state_data[:, 1], color='gray', linestyle='--', alpha=0.4, label="Telemetría (Ruido)")
                ax1.plot(t_num, state_pred[:, 1], color='#1f77b4', linewidth=3, label="Predicción PINN")
                ax1.set_xlabel("Tiempo (s)")
                ax1.set_ylabel("Altitud (m)")
                ax1.set_title("Ajuste de Trayectoria")
                ax1.legend(loc='upper right')
                
                # Panel Derecho: Loss Curve (Escala Logarítmica)
                ax2.plot(range(len(loss_history)), loss_history, color='#d62728', linewidth=1.5)
                ax2.set_yscale('log')
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Total Loss")
                ax2.set_title("Convergencia (Log Loss)")
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.92])
                
                # Guardar frame en buffer de memoria
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=120)
                buf.seek(0)
                frames.append(imageio.v2.imread(buf))
                plt.close(fig)
            
    print(f"\n🎬 Compilando animación final...")
    imageio.mimsave('docs/convergencia.gif', frames, fps=6, loop=0)
    print(f"✅ Animación guardada en docs/convergencia.gif ({len(frames)} frames)")
    
    # Guardar última imagen estática de alta calidad
    with torch.no_grad():
        state_final_pred = (model(t_data) * state_std + state_mean).cpu().numpy()
        t_final_np = t_data.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Altitud Final
    ax1.plot(t_final_np, state_data[:, 1], color='gray', alpha=0.3, label="Telemetría")
    ax1.plot(t_final_np, state_final_pred[:, 1], color='#1f77b4', linewidth=2.5, label="Predicción PINN")
    ax1.set_title("Ajuste Final de Altitud", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Altitud (m)")
    ax1.legend()

    # Velocidad Final
    ax2.plot(t_final_np, state_data[:, 3], color='gray', alpha=0.3)
    ax2.plot(t_final_np, state_final_pred[:, 3], color='#d62728', linewidth=2.5)
    ax2.set_title(f"Resultado Final: Cd = {model.cd.item():.3f} | Masa = {model.masa.item():.1f}kg", 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Velocidad Vertical (m/s)")
    
    plt.tight_layout()
    plt.savefig('docs/comparativa_pinn.png', dpi=300)
    print("📊 Gráfica final guardada en docs/comparativa_pinn.png")
