import numpy as np
import matplotlib.pyplot as plt
import math
import MicroMaglib as mml 

# ==========================================
# 1. PARÁMETROS FÍSICOS Y DE LA MALLA
# ==========================================
Ms = 800e3          
A = 13e-12          
K1 = 500.0          
uc_x, uc_y, uc_z = 1.0, 0.0, 0.0 

gamma0 = 2.211e5    
alpha = 0.5         
T = 300.0           

Nx, Ny, Nz = 100, 100, 1
dx, dy, dz = 5e-9, 5e-9, 5e-9

# ==========================================
# 2. INTERRUPTORES DE ENERGÍA
# ==========================================
usar_zeeman = True
usar_anisotropia = False
usar_intercambio = False
usar_desimanador = True

# =====================================================================
# 3. SIMULACIÓN CON ONDA SENOIDAL CONTINUA
# =====================================================================
print(f"--- INICIANDO SIMULACIÓN DINÁMICA (T = {T} K) ---")

H_max = 200e3
frecuencia = 100e6  # 100 MHz para permitir la inversión magnética
periodo = 1.0 / frecuencia

# Simulamos 1.25 periodos para asegurar: 
# (1) Primera imanación (2) Rama bajada (3) Rama subida completa
t_final = 1.25 * periodo  
dt = 1e-13
pasos_totales = int(t_final / dt)
intervalo_guardado = 10

# Estado inicial desimanado (aleatorio) para ver la primera imanación
m = np.random.uniform(-1, 1, (Nx, Ny, Nz, 3))
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            norm = np.linalg.norm(m[i, j, k])
            if norm > 0:
                m[i, j, k] = m[i, j, k] / norm

# Listas para guardar los datos
tiempos_lista = []
H_ext_lista = []
M_promedio_lista = []

# Pequeño ángulo para favorecer la nucleación y evitar estancamientos simétricos
angulo_inclinacion = math.radians(1.0)

for paso in range(pasos_totales):
    t = paso * dt
    
    # Onda senoidal proyectada
    H_mag = H_max * math.sin(2 * math.pi * frecuencia * t)
    H_ext_x = H_mag * math.cos(angulo_inclinacion)
    H_ext_y = H_mag * math.sin(angulo_inclinacion)
    H_ext_z = 0.0
    
    Heff_total = np.zeros_like(m)
    
    if usar_zeeman:
        Heff_total += mml.compute_zeeman_field_cpu(m, H_ext_x, H_ext_y, H_ext_z)
    if usar_anisotropia:
        Heff_total += mml.compute_anisotropy_field_cpu(m, K1, Ms, uc_x, uc_y, uc_z)
    if usar_intercambio:
        Heff_total += mml.compute_exchange_field(m, A, Ms, dx, dy, dz)
    if usar_desimanador:
        Heff_total += mml.compute_demag_field_direct_cpu(m, Ms, dx, dy, dz)

    # Integración de Heun con ruido térmico
    m = mml.heun_step_thermal(m, Heff_total, dt, gamma0, alpha, Ms, T, dx, dy, dz)
    
    # Guardamos los resultados cada 100 pasos para no saturar la memoria RAM
    if paso % intervalo_guardado == 0:
        tiempos_lista.append(t * 1e9)  # Guardado en nanosegundos
        H_ext_lista.append(H_mag)
        M_promedio_lista.append(np.mean(m[:, :, :, 0]))
        
    if paso % (pasos_totales // 10) == 0:
        print(f"Progreso: {100 * paso / pasos_totales:.1f}% (t = {t*1e9:.2f} ns)")

print("Simulación completada.")

# =====================================================================
# 4. ANÁLISIS NUMÉRICO DEL CICLO DE HISTÉRESIS
# =====================================================================
def analizar_ciclo(H_lista, M_lista, Ms, frec, dt, intervalo):
    """
    Analiza las listas de campo y magnetización para extraer los parámetros 
    físicos del bucle. Se analiza solo el último periodo para evitar transitorios.
    """
    # Convertimos a arrays de numpy
    H_arr = np.array(H_lista)
    M_arr = np.array(M_lista)
    
    # 1. Aislar el último periodo de oscilación
    pasos_por_periodo = int((1.0 / frec) / dt)
    puntos_guardados_periodo = pasos_por_periodo // intervalo # Guardamos cada 100 pasos
    
    H_loop = H_arr[-puntos_guardados_periodo:]
    M_loop = M_arr[-puntos_guardados_periodo:]
    
    # 2. Campo Coercitivo (H_c): Cruces de M por el cero
    cruces_M = np.where(np.diff(np.sign(M_loop)))[0]
    Hc_valores = []
    for idx in cruces_M:
        # Interpolación lineal para encontrar el H exacto donde M = 0
        H1, H2 = H_loop[idx], H_loop[idx+1]
        M1, M2 = M_loop[idx], M_loop[idx+1]
        Hc_interp = H1 - M1 * (H2 - H1) / (M2 - M1)
        Hc_valores.append(abs(Hc_interp))
    
    Hc_promedio = np.mean(Hc_valores) if len(Hc_valores) > 0 else 0.0
    
    # 3. Magnetización Remanente (M_r): Cruces de H por el cero
    cruces_H = np.where(np.diff(np.sign(H_loop)))[0]
    Mr_valores = []
    for idx in cruces_H:
        H1, H2 = H_loop[idx], H_loop[idx+1]
        M1, M2 = M_loop[idx], M_loop[idx+1]
        Mr_interp = M1 - H1 * (M2 - M1) / (H2 - H1)
        Mr_valores.append(abs(Mr_interp))
        
    Mr_promedio = np.mean(Mr_valores) if len(Mr_valores) > 0 else 0.0
    
    # 4. Área del Ciclo (Energía disipada W)
    # W = integral(mu0 * Ms * H * dM) -> Usamos el teorema de Green numérico
    mu0 = 4.0 * np.pi * 1e-7
    area_normalizada = np.abs(np.trapezoid(M_loop, H_loop)) 
    energia_disipada = mu0 * Ms * area_normalizada # Unidades: J/m^3
    
    # 5. Campo de Nucleación (H_n): Desviación del 2% desde la saturación
    # Buscamos en la rama descendente (cuando H va de Hmax a -Hmax)
    H_n = None
    for i in range(len(M_loop)):
        if H_loop[i] > 0 and np.gradient(H_loop)[i] < 0: # Aseguramos rama de bajada
            if M_loop[i] < 0.98: # Criterio del 2% de caída
                H_n = H_loop[i]
                break
    if H_n is None: H_n = Hc_promedio # Fallback
        
    # 6. Susceptibilidad diferencial máxima (chi_d)
    # Calculamos dM/dH. Evitamos divisiones por cero con diferencias finitas.
    dH = np.diff(H_loop)
    dM = np.diff(M_loop)
    # Filtramos los puntos donde dH es muy cercano a cero (picos de la onda senoidal)
    indices_validos = np.abs(dH) > 1.0 
    chi_d_array = np.abs(dM[indices_validos] / dH[indices_validos])
    chi_max = np.max(chi_d_array) if len(chi_d_array) > 0 else 0.0
    
    return Hc_promedio, Mr_promedio, energia_disipada, H_n, chi_max

# Llamada a la función de análisis
Hc, Mr, W, Hn, chi_max = analizar_ciclo(H_ext_lista, M_promedio_lista, Ms, frecuencia, dt, intervalo_guardado)

print("\n" + "="*50)
print("  RESULTADOS DEL ANÁLISIS DEL CICLO")
print("="*50)
print(f"  Campo Coercitivo (H_c) : {Hc/1000:.2f} kA/m")
print(f"  Mag. Remanente (m_r)   : {Mr:.4f} (Normalizada)")
print(f"  Energía Disipada (W)   : {W:.2e} J/m^3")
print(f"  Campo Nucleación (H_n) : {Hn/1000:.2f} kA/m")
print(f"  Susceptibilidad Max    : {chi_max:.2e} (Normalizada)")
print("="*50 + "\n")

# =====================================================================
# 5. GRÁFICAS DEL EXPERIMENTO
# =====================================================================
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfica 1: Espacio de Fases (Ciclo de Histéresis M vs H)
ax1.plot(H_ext_lista, M_promedio_lista, color='purple', linestyle='-', linewidth=2)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax1.set_title(f'Ciclo de Histéresis Dinámico ($f$ = {frecuencia/1e6:.0f} MHz)')
ax1.set_xlabel('Campo Externo $\mathbf{H}_x(t)$ (A/m)')
ax1.set_ylabel('Magnetización Promediada $\mathbf{m}_x(t)$')
ax1.set_xlim(-H_max*1.1, H_max*1.1)
ax1.set_ylim(-1.1, 1.1)

# Añadimos marcadores para el Hc en la gráfica
#ax1.plot([-Hc, Hc], [0, 0], 'ro', markersize=5, label='$\mathbf{H}_c$ numérico')
#ax1.legend()

# Gráfica 2: Dominio Temporal
H_norm = np.array(H_ext_lista) / H_max  

ax2.plot(tiempos_lista, H_norm, label='$\mathbf{H}_x(t)$ (Normalizado)', color='blue', linestyle='-.', linewidth=1.5)
ax2.plot(tiempos_lista, M_promedio_lista, label='$\mathbf{m}_x(t)$ (Promedio)', color='red', linewidth=2)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_title('Evolución Dinámica Temporal')
ax2.set_xlabel('Tiempo $t$ (ns)')
ax2.set_ylabel('Amplitud Normalizada')
ax2.legend(loc='lower left')
ax2.set_ylim(-1.2, 1.2)

plt.tight_layout()
plt.show()