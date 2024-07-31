import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import pandas as pd
import matplotlib.pyplot as plt

# Definir constantes
R = 0.0821  # Constante universal de los gases en L·atm/(mol·K)
P = 0.368  # Presión en atm (ajustada a atm si se usa R en L·atm/(mol·K))
T = 273 + 32  # Temperatura en K
Mw = 303.5  # Masa molar del solido 
Min = 58.8  # Masa molar del gas inerte 
ka = 0.00000861  # Coeficiente de transferencia 
kb = 0.0000059  # Coeficiente de transferencia 
kc = ka / kb  # Coeficiente de transferencia
MaxFlow = 0.01  # Flujo máximo en mol/s (condición inicial)

# Definir la altura de la capa seca
H = 0.001  # Valor placeholder en metros

# Definir la función ODE para el balance de masa
def mass_balance_ode(z, N):
    Nw, Nin = N
    dpdz = 0.0001  # Placeholder, reemplazar con el gradiente real si es conocido
    
    dNwdz = -(R / (P * Mw)) * (ka * dpdz + kb * P * dpdz)
    dNindz = -(R / (P * Min)) * (kc * dpdz + kb * P * dpdz)
    
    return [dNwdz, dNindz]

# Configurar el dominio
z_span = [0, H]

# Condiciones iniciales
N0 = [MaxFlow, MaxFlow]  # Flujos molares iniciales en z = 0 para ambos componentes

# Resolver la ODE para el balance de masa
sol_mass = solve_ivp(mass_balance_ode, z_span, N0, dense_output=True)

# Graficar resultados del balance de masa
z = np.linspace(0, H, 100)
N = sol_mass.sol(z)

# Encontrar los puntos máximos globales de los perfiles de flujo molar
Nw_max_idx = np.argmax(N[0])
Nin_max_idx = np.argmax(N[1])

Nw_max_z = z[Nw_max_idx]
Nin_max_z = z[Nin_max_idx]

Nw_max = N[0][Nw_max_idx]
Nin_max = N[1][Nin_max_idx]

# Imprimir los puntos óptimos globales en la consola
print(f'Punto óptimo global Nw: z = {Nw_max_z}, Nw = {Nw_max}')
print(f'Punto óptimo global Nin: z = {Nin_max_z}, Nin = {Nin_max}')

# Encontrar máximos y mínimos locales
Nw_local_max_idx = argrelextrema(N[0], np.greater, order=10)[0]
Nin_local_max_idx = argrelextrema(N[1], np.greater, order=10)[0]
Nw_local_min_idx = argrelextrema(N[0], np.less, order=10)[0]
Nin_local_min_idx = argrelextrema(N[1], np.less, order=10)[0]

# Seleccionar los 2 máximos locales más altos y los 2 mínimos locales más bajos
Nw_local_max_idx = Nw_local_max_idx[np.argsort(N[0][Nw_local_max_idx])][-2:]
Nin_local_max_idx = Nin_local_max_idx[np.argsort(N[1][Nin_local_max_idx])][-2:]
Nw_local_min_idx = Nw_local_min_idx[np.argsort(N[0][Nw_local_min_idx])][:2]
Nin_local_min_idx = Nin_local_min_idx[np.argsort(N[1][Nin_local_min_idx])][:2]

# Parámetros adicionales para el balance de energía
kII = 1.2  # Conductividad térmica de la capa congelada (valor placeholder)
rhoII = 9810  # Densidad de la capa congelada (valor placeholder)
cpII = 2000  # Calor específico de la capa congelada (valor placeholder)
hII_int = 8  # Coeficiente de transferencia de calor interno (valor placeholder)
Rg_int = 0.01  # Radio interno del vial (valor importante no mover de 0.01)
Tvial = 273 + 32  # Temperatura del vial

# Configuración del dominio espacial y temporal
L = 0.021  # Longitud total de la capa congelada
z_energy = np.linspace(0, L, 100)
t_energy = np.linspace(0, 8*3600, 100)  # Simular por 8 horas

# Condiciones iniciales
T0 = np.ones_like(z_energy) * 303  # Temperatura inicial uniforme (ejemplo)

# Función para el balance de energía en la capa congelada
def energy_balance_frozen(t, T):
    dTdz2 = np.gradient(np.gradient(T, z_energy), z_energy)
    dTdt = (kII / (rhoII * cpII)) * dTdz2 + (hII_int * 2 / (rhoII * cpII * Rg_int)) * (Tvial - T)
    return dTdt

# Resolver la PDE usando solve_ivp
sol_energy = solve_ivp(energy_balance_frozen, [t_energy[0], t_energy[-1]], T0, t_eval=t_energy, method='RK45')

# Resultados del balance de energía
T = sol_energy.y

# Encontrar el punto máximo global de temperatura
T_max_idx = np.unravel_index(np.argmax(T), T.shape)
T_max_z = z_energy[T_max_idx[0]]
T_max_t = t_energy[T_max_idx[1]]

T_max = T[T_max_idx]

# Imprimir el punto óptimo global en la consola
print(f'Punto óptimo global de temperatura: z = {T_max_z}, t = {T_max_t}, T = {T_max}')

# Encontrar puntos máximos y mínimos locales de temperatura
local_max_idx = argrelextrema(T, np.greater, order=10, axis=1)
local_min_idx = argrelextrema(T, np.less, order=10, axis=1)

# Seleccionar los 2 máximos locales más altos y los 2 mínimos locales más bajos
top2_local_max_idx = np.argsort(T[local_max_idx])[::-1][:2]
top2_local_min_idx = np.argsort(T[local_min_idx])[:2]

# Crear tabla de resultados combinada
results_combined = {
    'z': np.tile(z_energy, len(t_energy)),
    't': np.repeat(t_energy, len(z_energy)),
    'Nw': np.repeat(N[0], len(t_energy)),
    'Nin': np.repeat(N[1], len(t_energy)),
    'T': T.flatten()
}
df_combined = pd.DataFrame(results_combined)
df_combined.to_excel('combined_results.xlsx', index=False)

# Plot del elemento diferencial (cubo) con el fenómeno representado
fig_cube = plt.figure()
ax_cube = fig_cube.add_subplot(111, projection='3d')

# Crear un cubo de ejemplo con los resultados del balance de masa y energía
X, Y = np.meshgrid(np.linspace(0, L, 10), np.linspace(0, H, 10))
Z = np.zeros_like(X)

# Representar el fenómeno en el cubo con flechas de flujo
ax_cube.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(T[:, 0].reshape(X.shape) / T.max()), rstride=1, cstride=1, alpha=0.9)

# Ajustar la posición y tamaño de los vectores de flujo
for i in range(len(X)):
    for j in range(len(Y)):
        ax_cube.quiver(X[i, j], Y[i, j], Z[i, j], 0, 0, N[0][j] * 1e5, color='blue', alpha=0.6, length=0.0125, normalize=True)

# Añadir una barra de color para la escala de temperatura
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(T.flatten())
fig_cube.colorbar(mappable, ax=ax_cube, label='Temperatura (K)')

ax_cube.set_xlabel('Longitud (m)')
ax_cube.set_ylabel('Altura (m)')
ax_cube.set_zlabel('Posición (m)')
ax_cube.set_title('Representación del fenómeno en el elemento diferencial (cubo)')

plt.show()

