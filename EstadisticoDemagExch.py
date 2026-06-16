import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import MicroMaglib as mml

def ejecutar_simulacion_estadistica():
    # --- Parámetros Físicos y Numéricos ---
    # Valores típicos (ej. Permalloy). Ajusta según tu material.
    Ms = 8.0e5                 # Magnetización de saturación (A/m)
    A = 1.3e-11                # Constante de intercambio (J/m)
    alpha = 0.5                # Amortiguamiento alto para acelerar relajación
    gamma0 = 2.21e5           # Ratio giromagnético (m/A·s) - Signo negativo explícito
    
    # Parámetros térmicos y temporales
    T_simulacion = 300.0       # Temperatura para la agitación térmica (K)
    dt = 1e-13                 # Paso de tiempo (s)
    pasos_termicos = 1000      # Pasos de evolución con T > 0
    pasos_relajacion = 500     # Pasos de relajación con T = 0
    N_realizaciones = 100      # Estadísticas por tamaño
    
    # Geometría
    tam_celda = 2.5e-9           # Resolución de la malla discreta (m)
    dx = dy = dz = tam_celda
    Nz = 1
    
    # Tamaños L a simular en nm
    L_min = 20       # Tamaño inicial en nm
    L_max = 160      # Tamaño final en nm
    paso_L = 10      # Incremento en nm

    # np.arange no incluye el límite superior por defecto, por eso sumamos un poco (o el paso entero)
    L_array_nm = np.arange(L_min, L_max + 1, paso_L)
    
    resultados = []

    print("--- Iniciando Simulación Micromagnética en CPU ---")
    
    for L_nm in L_array_nm:
        L = L_nm * 1e-9
        Nx = int(L / dx)
        Ny = int(L / dy)
        
        print(f"\nSimulando L = {L_nm} nm (Malla: {Nx}x{Ny}x{Nz})")
        m_finales = []
        
        for realizacion in range(N_realizaciones):
            # 3) Arranque completamente aleatorio
            m = np.random.uniform(-1, 1, (Nx, Ny, Nz, 3))
            norma = np.linalg.norm(m, axis=-1, keepdims=True)
            m = m / norma
            
            # 4.a) Fase Térmica: Exploración del espacio de estados
            for paso in range(pasos_termicos):
                # Calcular campos efectivos espaciales
                H_ex = mml.compute_exchange_field(m, A, Ms, dx, dy, dz)
                H_demag = mml.compute_demag_field_direct_cpu(m, Ms, dx, dy, dz)
                
                # Campo efectivo total Heff
                H_eff = H_ex + H_demag
                
                # Evolución de Heun con T > 0
                m = mml.heun_step_thermal(m, H_eff, dt, gamma0, alpha, Ms, T_simulacion, dx, dy, dz)
            
            # 4.b) Fase de Relajación: Caída al estado base (T = 0)
            for paso in range(pasos_relajacion):
                H_ex = mml.compute_exchange_field(m, A, Ms, dx, dy, dz)
                H_demag = mml.compute_demag_field_direct_cpu(m, Ms, dx, dy, dz)
                H_eff = H_ex + H_demag
                
                # Reutilizamos heun_step_thermal pero con T = 0 para simular la relajación RK determinista
                m = mml.heun_step_thermal(m, H_eff, dt, gamma0, alpha, Ms, 0.0, dx, dy, dz)
            
            # 5) Cálculo de la magnetización remanente promedio |m|
            # Promedio espacial de las componentes
            m_promedio_vector = np.mean(m, axis=(0, 1, 2))
            # Módulo del vector promedio
            modulo_m = np.linalg.norm(m_promedio_vector)
            m_finales.append(modulo_m)
            
        # Estadísticas para el tamaño L actual
        media_m = np.mean(m_finales)
        std_m = np.std(m_finales)
        
        resultados.append({
            'L_nm': L_nm,
            'Media_m': media_m,
            'Std_m': std_m
        })
        print(f"  Resultado -> Media |m|: {media_m:.3f} ± {std_m:.3f}")

    # 6) Guardado de datos en un archivo CSV
    df = pd.DataFrame(resultados)
    df.to_csv("datos_transicion_dominios.csv", index=False)
    print("\nSimulación completada. Datos guardados en 'datos_transicion_dominios.csv'.")

def graficar_resultados(archivo_csv):
    # Comprobar si el archivo existe
    if not os.path.exists(archivo_csv):
        print(f"Error: No se encuentra el archivo {archivo_csv}")
        return

    # Leer los datos guardados
    df = pd.DataFrame(pd.read_csv(archivo_csv))
    L = df['L_nm']
    media = df['Media_m']
    desviacion = df['Std_m']

    # --- Configuración del Gráfico ---
    plt.figure(figsize=(10, 6))
    
    # Dibujar línea horizontal de saturación ideal |m|=1 y |m|=0
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1.2)
    plt.axhline(0.0, color='black', linestyle='-', linewidth=1.2)

    # Añadir regiones sombreadas de fondo (Regímenes)
    plt.axvspan(min(L)-5, 55, facecolor='#ffe6e6', alpha=0.5, label='Régimen de Monodominio')
    plt.axvspan(55, max(L)+5, facecolor='#e6f2e6', alpha=0.5, label='Régimen Multidominio')

    # Graficar los puntos con barras de error
    plt.errorbar(L, media, yerr=desviacion, fmt='-o', color='#1f77b4', 
                 ecolor='#d62728', elinewidth=2, capsize=5, capthick=2, 
                 markersize=8, linewidth=2.5, label=r'Media y Desviación (N=100)')

    # Textos anotativos
    plt.text(35, 0.45, "Monodominio\n(Intercambio dominante)", 
             color='#b30000', fontsize=12, fontweight='bold', ha='center')
    plt.text(105, 0.45, "Vórtices y Dominios\n(Dipolar dominante)", 
             color='#1a661a', fontsize=12, fontweight='bold', ha='center')

    # Ajustes visuales de ejes y leyendas
    plt.xlabel(r'Lado de la partícula cuadrada $L$ (nm)', fontsize=14)
    plt.ylabel(r'Magnetización Remanente Promedio $|\mathbf{m}|$', fontsize=14)
    plt.title('Transición de Estado Base: Análisis Estadístico', fontsize=16, fontweight='bold', pad=15)
    
    plt.xlim(min(L)-5, max(L)+5)
    plt.ylim(-0.05, 1.1)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Mostrar o guardar el gráfico
    plt.savefig("Grafica_Transicion_Generada.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Comentar o descomentar según lo que se necesite ejecutar
    
    # Paso 1: Ejecutar simulaciones y crear el .csv
    ejecutar_simulacion_estadistica()
    
    # Paso 2: Leer el .csv y generar el gráfico
    graficar_resultados("datos_transicion_dominios.csv")