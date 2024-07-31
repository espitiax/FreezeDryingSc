import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema

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

# Crear figura para el balance de masa
fig_mass = plt.figure()
ax_mass = fig_mass.add_subplot(111, projection='3d')
Z_mass, N_mesh = np.meshgrid(z, [0, 1])
ax_mass.plot_surface(Z_mass, N_mesh, np.array([N[0], N[1]]), cmap='viridis')

# Añadir puntos óptimos globales a la gráfica
ax_mass.scatter(Nw_max_z, 0, Nw_max, color='r', s=100, label='Nw máximo global')
ax_mass.scatter(Nin_max_z, 1, Nin_max, color='b', s=100, label='Nin máximo global')

# Añadir puntos óptimos locales a la gráfica
ax_mass.scatter(z[Nw_local_max_idx], [0] * len(Nw_local_max_idx), N[0][Nw_local_max_idx], color='r', s=50, label='Nw máximo local')
ax_mass.scatter(z[Nin_local_max_idx], [1] * len(Nin_local_max_idx), N[1][Nin_local_max_idx], color='b', s=50, label='Nin máximo local')
ax_mass.scatter(z[Nw_local_min_idx], [0] * len(Nw_local_min_idx), N[0][Nw_local_min_idx], color='g', s=50, label='Nw mínimo local')
ax_mass.scatter(z[Nin_local_min_idx], [1] * len(Nin_local_min_idx), N[1][Nin_local_min_idx], color='y', s=50, label='Nin mínimo local')

ax_mass.set_xlabel('z (m)')
ax_mass.set_ylabel('Tiempo (s)')
ax_mass.set_zlabel('Flujo molar (mol/s)')
ax_mass.set_title('Perfiles de flujo molar en la capa seca')
ax_mass.legend()
plt.colorbar(ax_mass.plot_surface(Z_mass, N_mesh, np.array([N[0], N[1]]), cmap='viridis'), ax=ax_mass, shrink=0.5, aspect=5)

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
T0 = np.ones_like(z_energy) *303  # Temperatura inicial uniforme (ejemplo)

# Función para el balance de energía en la capa congelada
def energy_balance_frozen(t, T):
    dTdz2 = np.gradient(np.gradient(T, z_energy), z_energy)
    dTdt = (kII / (rhoII * cpII)) * dTdz2 + (hII_int * 2 / (rhoII * cpII * Rg_int)) * (Tvial - T)
    return dTdt

# Resolver la PDE usando solve_ivp
sol_energy = solve_ivp(energy_balance_frozen, [t_energy[0], t_energy[-1]], T0, t_eval=t_energy, method='RK45')

# Graficar resultados
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

fig_energy = plt.figure()
ax_energy = fig_energy.add_subplot(111, projection='3d')
Z_energy, T_mesh = np.meshgrid(z_energy, t_energy)
ax_energy.plot_surface(Z_energy, T_mesh, T.T, cmap='viridis')

# Añadir punto óptimo global a la gráfica
ax_energy.scatter(T_max_z, T_max_t, T_max, color='r', s=100, label='T máximo global')

# Añadir puntos óptimos locales a la gráfica
for i in top2_local_max_idx:
    z_idx = local_max_idx[1][i]
        # Continuación de la adición de puntos óptimos locales a la gráfica de temperatura
    t_idx = local_max_idx[0][i]
    local_max_z = z_energy[z_idx]
    local_max_t = t_energy[t_idx]
    local_max = T[z_idx, t_idx]
    ax_energy.scatter(local_max_z, local_max_t, local_max, color='g', s=50, label='T máximo local')

for i in top2_local_min_idx:
    z_idx = local_min_idx[1][i]
    t_idx = local_min_idx[0][i]
    local_min_z = z_energy[z_idx]
    local_min_t = t_energy[t_idx]
    local_min = T[z_idx, t_idx]
    ax_energy.scatter(local_min_z, local_min_t, local_min, color='y', s=50, label='T mínimo local')

ax_energy.set_xlabel('Posición (m)')
ax_energy.set_ylabel('Tiempo (s)')
ax_energy.set_zlabel('Temperatura (K)')
ax_energy.set_title('Perfil de temperatura en la capa congelada')
ax_energy.legend()
plt.colorbar(ax_energy.plot_surface(Z_energy, T_mesh, T.T, cmap='viridis'), ax=ax_energy, shrink=0.5, aspect=5)

plt.show()

