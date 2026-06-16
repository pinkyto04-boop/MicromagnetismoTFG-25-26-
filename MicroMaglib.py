

#Esta será la librería central del las simulaciones micromagnéticas 
import numpy as np
from numba import cuda
from numba import njit, prange
import math


"""
    Calcula el campo efectivo de intercambio en una malla 3D usando Numba.
    
    Parámetros:
    m  : array de numpy de dimensiones (Nx, Ny, Nz, 3) con |m| = 1
    A  : Constante de rigidez de intercambio (J/m)
    Ms : Magnetización de saturación (A/m)
    dx, dy, dz : Tamaños de celda en metros
    
    Retorna:
    H_ex : array de numpy de dimensiones (Nx, Ny, Nz, 3) con el campo en A/m
    """
@njit(parallel=True)
def compute_exchange_field(m, A, Ms, dx, dy, dz):
   
    Nx, Ny, Nz, _ = m.shape
    H_ex = np.zeros_like(m)
    
    # Constante universal y prefactor
    mu0 = 4 * np.pi * 1e-7
    coef = (2 * A) / (mu0 * Ms)
    
    # Cuadrado de los tamaños de celda para el denominador del laplaciano
    dx2 = dx**2
    dy2 = dy**2
    dz2 = dz**2

    # prange en el bucle exterior divide la malla 3D en bloques a lo largo del eje X,
    # repartiéndolos entre los distintos núcleos de la CPU de forma eficiente.
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                
                # Evaluamos de forma independiente las componentes mx, my, mz
                for c in range(3):
                    
                    # --- Derivada en X (Condición de contorno Neumann en bordes) ---
                    # ¿Existe un nodo anterior? Si no (porque i=0), copia el nodo actual m[0]
                    m_prev_x = m[i-1, j, k, c] if i > 0 else m[i, j, k, c]
                    # ¿Existe un nodo siguiente? Si no (estamos en el límite superior Nx-1), copia el nodo actual
                    m_next_x = m[i+1, j, k, c] if i < Nx - 1 else m[i, j, k, c]
                    d2m_dx2 = (m_next_x - 2.0 * m[i, j, k, c] + m_prev_x) / dx2
                    
                    # --- Derivada en Y (Condición de contorno Neumann en bordes) ---
                    # ¿Existe un nodo anterior? Si no (porque i=0), copia el nodo actual m[0]
                    m_prev_y = m[i, j-1, k, c] if j > 0 else m[i, j, k, c]
                    # ¿Existe un nodo siguiente? Si no (estamos en el límite superior Nx-1), copia el nodo actual
                    m_next_y = m[i, j+1, k, c] if j < Ny - 1 else m[i, j, k, c]
                    d2m_dy2 = (m_next_y - 2.0 * m[i, j, k, c] + m_prev_y) / dy2
                    
                    # --- Derivada en Z (Condición de contorno Neumann en bordes) ---
                    # ¿Existe un nodo anterior? Si no (porque i=0), copia el nodo actual m[0]
                    m_prev_z = m[i, j, k-1, c] if k > 0 else m[i, j, k, c]
                    # ¿Existe un nodo siguiente? Si no (estamos en el límite superior Nx-1), copia el nodo actual
                    m_next_z = m[i, j, k+1, c] if k < Nz - 1 else m[i, j, k, c]
                    d2m_dz2 = (m_next_z - 2.0 * m[i, j, k, c] + m_prev_z) / dz2
                    
                    # Superposición para obtener el Laplaciano total y el campo final
                    H_ex[i, j, k, c] = coef * (d2m_dx2 + d2m_dy2 + d2m_dz2)
                    
    return H_ex


"""
    Calcula el campo de anisotropía uniaxial en la CPU usando múltiples núcleos.
    
    Parámetros:
    m    : array de numpy (Nx, Ny, Nz, 3) con la magnetización normalizada.
    K1   : Constante de anisotropía de primer orden (J/m^3).
    Ms   : Magnetización de saturación (A/m).
    uc_x, uc_y, uc_z : Componentes del vector unitario del eje fácil.
    
    Retorna:
    H_ani : array de numpy (Nx, Ny, Nz, 3) con el campo efectivo en A/m.
    """

@njit(parallel=True)
def compute_anisotropy_field_cpu(m, K1, Ms, uc_x, uc_y, uc_z):
    
    Nx, Ny, Nz, _ = m.shape
    H_ani = np.zeros_like(m)
    
    mu0 = 4.0 * math.pi * 1e-7
    
    # Precalculamos el factor constante fuera de los bucles para ahorrar operaciones
    prefactor = (2.0 * K1) / (mu0 * Ms)
    
    # prange le dice a Numba que divida este bucle entre los distintos núcleos de la CPU
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                
                # Extraemos las componentes locales
                mx = m[i, j, k, 0]
                my = m[i, j, k, 1]
                mz = m[i, j, k, 2]
                
                # Producto escalar de la magnetización local con el eje de fácil imanación
                m_dot_uc = (mx * uc_x) + (my * uc_y) + (mz * uc_z)
                
                # Coeficiente total de la celda
                coef = prefactor * m_dot_uc
                
                # Asignamos el campo apuntando en la dirección del eje fácil
                H_ani[i, j, k, 0] = coef * uc_x
                H_ani[i, j, k, 1] = coef * uc_y
                H_ani[i, j, k, 2] = coef * uc_z
                
    return H_ani


"""
    Calcula el campo de Zeeman estático uniforme en la CPU.
    
    Parámetros:
    m : array (Nx, Ny, Nz, 3) con la magnetización (usado solo para sacar las dimensiones).
    H_ext_x, H_ext_y, H_ext_z : Componentes del campo magnético externo aplicado (A/m).
    
    Retorna:
    H_zeeman : array (Nx, Ny, Nz, 3) con el campo efectivo en A/m.
    """
@njit(parallel=True)
def compute_zeeman_field_cpu(m, H_ext_x, H_ext_y, H_ext_z):
    """
    Calcula el campo de Zeeman estático uniforme en la CPU.
    
    Parámetros:
    m : array (Nx, Ny, Nz, 3) con la magnetización (usado solo para sacar las dimensiones).
    H_ext_x, H_ext_y, H_ext_z : Componentes del campo magnético externo aplicado (A/m).
    
    Retorna:
    H_zeeman : array (Nx, Ny, Nz, 3) con el campo efectivo en A/m.
    """
    Nx, Ny, Nz, _ = m.shape
    H_zeeman = np.zeros_like(m)
    
    # prange divide la asignación de memoria entre los núcleos disponibles
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # El campo es el mismo para absolutamente todos los nodos
                H_zeeman[i, j, k, 0] = H_ext_x
                H_zeeman[i, j, k, 1] = H_ext_y
                H_zeeman[i, j, k, 2] = H_ext_z
                
    return H_zeeman




"""
    Calcula el campo desmagnetizante mediante la suma directa de dipolos.
    """

@njit(parallel=True)
def compute_demag_field_direct_cpu(m, Ms, dx, dy, dz):
   
    Nx, Ny, Nz, _ = m.shape
    H_demag = np.zeros_like(m)
    
    # Volumen de la celda
    V_cell = dx * dy * dz
    
    # Prefactor que agrupa todas las constantes del dipolo
    prefactor = (Ms * V_cell) / (4.0 * math.pi)
    
    # Recorremos todas las celdas DESTINO (las que sienten el campo)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                
                hx_total = 0.0
                hy_total = 0.0
                hz_total = 0.0
                
                # Recorremos todas las celdas ORIGEN (las que generan el campo)
                for i2 in range(Nx):
                    for j2 in range(Ny):
                        for k2 in range(Nz):
                            
                            # Una celda puntual no genera este campo sobre sí misma
                            if i == i2 and j == j2 and k == k2:
                                continue
                            
                            # Componentes del vector distancia r_ij
                            rx = (i - i2) * dx
                            ry = (j - j2) * dy
                            rz = (k - k2) * dz
                            
                            r2 = rx**2 + ry**2 + rz**2
                            r = math.sqrt(r2)
                            r5 = r2 * r2 * r # r^5 para el denominador
                            
                            # Magnetización de la celda origen
                            mx = m[i2, j2, k2, 0]
                            my = m[i2, j2, k2, 1]
                            mz = m[i2, j2, k2, 2]
                            
                            # Producto escalar (m · r_ij)
                            m_dot_r = mx * rx + my * ry + mz * rz
                            
                            # Superposición de la contribución del dipolo
                            hx_total += (3.0 * m_dot_r * rx - mx * r2) / r5
                            hy_total += (3.0 * m_dot_r * ry - my * r2) / r5
                            hz_total += (3.0 * m_dot_r * rz - mz * r2) / r5
                            
                # Multiplicamos por las constantes y guardamos el campo final
                H_demag[i, j, k, 0] = prefactor * hx_total
                H_demag[i, j, k, 1] = prefactor * hy_total
                H_demag[i, j, k, 2] = prefactor * hz_total
                
    return H_demag

"""
    Calcula dm/dt para un único espín m_vec (array de 3 elementos) 
    sujeto a un campo efectivo Heff_vec.
    """

@njit
def compute_llg_torque(m_vec, Heff_vec, gamma0, alpha):
    
    # Prefactores de la ecuación LLG explícita
    prefactor_prec = gamma0 / (1.0 + alpha**2)
    prefactor_damp = (alpha * gamma0) / (1.0 + alpha**2)
    
    # Precesión: m x Heff
    px = m_vec[1] * Heff_vec[2] - m_vec[2] * Heff_vec[1]
    py = m_vec[2] * Heff_vec[0] - m_vec[0] * Heff_vec[2]
    pz = m_vec[0] * Heff_vec[1] - m_vec[1] * Heff_vec[0]
    
    # Amortiguamiento: m x (m x Heff) = m x p
    dx = m_vec[1] * pz - m_vec[2] * py
    dy = m_vec[2] * px - m_vec[0] * pz
    dz = m_vec[0] * py - m_vec[1] * px
    
    # Torque total dm/dt
    dm_dt_x = prefactor_prec * px - prefactor_damp * dx
    dm_dt_y = prefactor_prec * py - prefactor_damp * dy
    dm_dt_z = prefactor_prec * pz - prefactor_damp * dz
    
    return dm_dt_x, dm_dt_y, dm_dt_z

"""
    Avanza la magnetización un paso dt usando Runge-Kutta de 4to orden.
    Nota: Heff aquí debe ser el campo total evaluado en el instante t.
    """

@njit(parallel=True)
def rk4_step(m, Heff, dt, gamma0, alpha):
    
    Nx, Ny, Nz, _ = m.shape
    m_new = np.zeros_like(m)
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                m_curr = m[i, j, k]
                H_curr = Heff[i, j, k]
                
                # k1: Evaluado al inicio del intervalo
                k1_x, k1_y, k1_z = compute_llg_torque(m_curr, H_curr, gamma0, alpha)
                
                # k2: Evaluado en el punto medio usando k1
                m2 = np.array([m_curr[0] + 0.5*dt*k1_x, 
                               m_curr[1] + 0.5*dt*k1_y, 
                               m_curr[2] + 0.5*dt*k1_z])
                k2_x, k2_y, k2_z = compute_llg_torque(m2, H_curr, gamma0, alpha)
                
                # k3: Evaluado en el punto medio usando k2
                m3 = np.array([m_curr[0] + 0.5*dt*k2_x, 
                               m_curr[1] + 0.5*dt*k2_y, 
                               m_curr[2] + 0.5*dt*k2_z])
                k3_x, k3_y, k3_z = compute_llg_torque(m3, H_curr, gamma0, alpha)
                
                # k4: Evaluado al final del intervalo usando k3
                m4 = np.array([m_curr[0] + dt*k3_x, 
                               m_curr[1] + dt*k3_y, 
                               m_curr[2] + dt*k3_z])
                k4_x, k4_y, k4_z = compute_llg_torque(m4, H_curr, gamma0, alpha)
                
                # Combinación final RK4
                m_new_x = m_curr[0] + (dt/6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
                m_new_y = m_curr[1] + (dt/6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
                m_new_z = m_curr[2] + (dt/6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z)
                
                # Normalización fuerte para asegurar |m| = 1
                norm = math.sqrt(m_new_x**2 + m_new_y**2 + m_new_z**2)
                m_new[i, j, k, 0] = m_new_x / norm
                m_new[i, j, k, 1] = m_new_y / norm
                m_new[i, j, k, 2] = m_new_z / norm
                
    return m_new



@njit(parallel=True)
def heun_step_thermal(m, Heff, dt, gamma0, alpha, Ms, T, dx, dy, dz):
    """
    Avanza la magnetización usando el método de Heun para SDEs (Stratonovich),
    incorporando un campo de ruido térmico blanco y gaussiano.
    """
    Nx, Ny, Nz, _ = m.shape
    m_new = np.zeros_like(m)
    
    kB = 1.380649e-23  # Constante de Boltzmann (J/K)
    V_cell = dx * dy * dz
    
    # Varianza del campo térmico según fluctuación-disipación
    # Se añade mu0 para compatibilidad de unidades A/m
    if T > 0.0:
        std_dev_th = math.sqrt((2.0 * alpha * kB * T) / (gamma0 * Ms * V_cell * dt))
    else:
        std_dev_th = 0.0

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                m_curr = m[i, j, k]
                H_det = Heff[i, j, k]
                
                # Generamos el campo estocástico (h_fl) una sola vez por paso y celda
                if T > 0.0:
                    h_th = np.array([np.random.normal(0.0, std_dev_th),
                                     np.random.normal(0.0, std_dev_th),
                                     np.random.normal(0.0, std_dev_th)])
                else:
                    h_th = np.array([0.0, 0.0, 0.0])
                
                # Campo total = Determinista + Térmico
                H_tot = np.array([H_det[0] + h_th[0], 
                                  H_det[1] + h_th[1], 
                                  H_det[2] + h_th[2]])
                
                # 1. PREDICTOR (Paso de Euler)
                dm_dt_P_x, dm_dt_P_y, dm_dt_P_z = compute_llg_torque(m_curr, H_tot, gamma0, alpha)
                
                m_pred = np.array([m_curr[0] + dt * dm_dt_P_x,
                                   m_curr[1] + dt * dm_dt_P_y,
                                   m_curr[2] + dt * dm_dt_P_z])
                
                # 2. CORRECTOR (Evaluado en m_prediccion pero con EL MISMO H_tot)
                dm_dt_C_x, dm_dt_C_y, dm_dt_C_z = compute_llg_torque(m_pred, H_tot, gamma0, alpha)
                
                # Actualización final (Promedio de torques)
                m_new_x = m_curr[0] + 0.5 * dt * (dm_dt_P_x + dm_dt_C_x)
                m_new_y = m_curr[1] + 0.5 * dt * (dm_dt_P_y + dm_dt_C_y)
                m_new_z = m_curr[2] + 0.5 * dt * (dm_dt_P_z + dm_dt_C_z)
                
                # Normalización fuerte
                norm = math.sqrt(m_new_x**2 + m_new_y**2 + m_new_z**2)
                m_new[i, j, k, 0] = m_new_x / norm
                m_new[i, j, k, 1] = m_new_y / norm
                m_new[i, j, k, 2] = m_new_z / norm
                
    return m_new



###########################################################################################

################ FUNCIONES PARA CORRER EL PROGRAMA EN LA GPU ##############################

###########################################################################################





@cuda.jit
def compute_exchange_field_cuda(m, H_ex, A, Ms, dx, dy, dz):
    """
    Kernel de CUDA para calcular el campo de intercambio.
    A diferencia de CPU, los kernels no devuelven arrays (no hay 'return'), 
    sino que modifican el array de salida 'H_ex' directamente en la memoria de la GPU.
    """
    # 1. Obtener el índice absoluto del hilo en el espacio 3D
    i, j, k = cuda.grid(3)
    
    Nx, Ny, Nz, _ = m.shape
    
    # 2. Asegurarnos de que el hilo actual no se sale de los límites de la malla
    if i < Nx and j < Ny and k < Nz:
        
        mu0 = 4.0 * np.pi * 1e-7
        coef = (2.0 * A) / (mu0 * Ms)
        
        dx2 = dx**2
        dy2 = dy**2
        dz2 = dz**2
        
        # 3. Iteramos solo sobre las 3 componentes del vector (x, y, z)
        for c in range(3):
            # --- Derivada en X ---
            m_prev_x = m[i-1, j, k, c] if i > 0 else m[i, j, k, c]
            m_next_x = m[i+1, j, k, c] if i < Nx - 1 else m[i, j, k, c]
            d2m_dx2 = (m_next_x - 2.0 * m[i, j, k, c] + m_prev_x) / dx2
            
            # --- Derivada en Y ---
            m_prev_y = m[i, j-1, k, c] if j > 0 else m[i, j, k, c]
            m_next_y = m[i, j+1, k, c] if j < Ny - 1 else m[i, j, k, c]
            d2m_dy2 = (m_next_y - 2.0 * m[i, j, k, c] + m_prev_y) / dy2
            
            # --- Derivada en Z ---
            m_prev_z = m[i, j, k-1, c] if k > 0 else m[i, j, k, c]
            m_next_z = m[i, j, k+1, c] if k < Nz - 1 else m[i, j, k, c]
            d2m_dz2 = (m_next_z - 2.0 * m[i, j, k, c] + m_prev_z) / dz2
            
            # 4. Escribir el resultado directamente en el array de la GPU
            H_ex[i, j, k, c] = coef * (d2m_dx2 + d2m_dy2 + d2m_dz2)



@cuda.jit
def compute_anisotropy_field_cuda(m, H_ani, K1, Ms, uc_x, uc_y, uc_z):
    """
    Kernel de CUDA para calcular el campo de anisotropía uniaxial.
    """
    # 1. Obtener el índice absoluto del hilo en el espacio 3D
    i, j, k = cuda.grid(3)
    
    Nx, Ny, Nz, _ = m.shape
    
    # 2. Asegurarnos de que el hilo actual no se sale de los límites
    if i < Nx and j < Ny and k < Nz:
        
        mu0 = 4.0 * math.pi * 1e-7
        
        # 3. Extraer las componentes locales de la magnetización
        mx = m[i, j, k, 0]
        my = m[i, j, k, 1]
        mz = m[i, j, k, 2]
        
        # 4. Calcular el producto escalar (m · uc)
        m_dot_uc = (mx * uc_x) + (my * uc_y) + (mz * uc_z)
        
        # 5. Calcular el prefactor escalar: 2*K1 / (mu0 * Ms) * (m · uc)
        coef = (2.0 * K1) / (mu0 * Ms) * m_dot_uc
        
        # 6. Escribir el campo resultante directamente en VRAM
        H_ani[i, j, k, 0] = coef * uc_x
        H_ani[i, j, k, 1] = coef * uc_y
        H_ani[i, j, k, 2] = coef * uc_z



def compute_zeeman_field_cuda(m, H_zeeman, H_ext_x, H_ext_y, H_ext_z):
    """
    Kernel de CUDA para el campo de Zeeman estático uniforme.
    """
    # Identificador único del hilo en la malla 3D
    i, j, k = cuda.grid(3)
    
    Nx, Ny, Nz, _ = m.shape
    
    # Verificación de límites
    if i < Nx and j < Ny and k < Nz:
        H_zeeman[i, j, k, 0] = H_ext_x
        H_zeeman[i, j, k, 1] = H_ext_y
        H_zeeman[i, j, k, 2] = H_ext_z


@cuda.jit
def compute_demag_field_direct_cuda(m, H_demag, Ms, dx, dy, dz):
    """
    Kernel de CUDA para el campo desmagnetizante (Suma directa dipolo-dipolo).
    """
    # 1. Identificamos la celda DESTINO (la posición de este hilo en la malla)
    i, j, k = cuda.grid(3)
    
    Nx, Ny, Nz, _ = m.shape
    
    # 2. Comprobamos límites
    if i < Nx and j < Ny and k < Nz:
        
        # Volumen y prefactor (constantes para este hilo)
        V_cell = dx * dy * dz
        prefactor = (Ms * V_cell) / (4.0 * math.pi)
        
        hx_total = 0.0
        hy_total = 0.0
        hz_total = 0.0
        
        # 3. El hilo recorre TODA la malla buscando las celdas ORIGEN
        for i2 in range(Nx):
            for j2 in range(Ny):
                for k2 in range(Nz):
                    
                    # Evitamos la auto-interacción (división por cero)
                    if i == i2 and j == j2 and k == k2:
                        continue
                    
                    # Vector distancia r_ij
                    rx = (i - i2) * dx
                    ry = (j - j2) * dy
                    rz = (k - k2) * dz
                    
                    r2 = rx**2 + ry**2 + rz**2
                    r = math.sqrt(r2)
                    r5 = r2 * r2 * r
                    
                    # Leer magnetización de la celda origen
                    mx = m[i2, j2, k2, 0]
                    my = m[i2, j2, k2, 1]
                    mz = m[i2, j2, k2, 2]
                    
                    m_dot_r = mx * rx + my * ry + mz * rz
                    
                    # Sumar la contribución de este dipolo al acumulador local
                    hx_total += (3.0 * m_dot_r * rx - mx * r2) / r5
                    hy_total += (3.0 * m_dot_r * ry - my * r2) / r5
                    hz_total += (3.0 * m_dot_r * rz - mz * r2) / r5
                    
        # 4. Escribir el resultado final en la memoria de la GPU
        H_demag[i, j, k, 0] = prefactor * hx_total
        H_demag[i, j, k, 1] = prefactor * hy_total
        H_demag[i, j, k, 2] = prefactor * hz_total



@cuda.jit(device=True)
def compute_llg_torque_cuda(mx, my, mz, Hx, Hy, Hz, gamma0, alpha):
    """
    Calcula dm/dt para un único espín. 
    Al ser 'device=True', esta función se ejecuta de forma privada en cada hilo.
    """
    prefactor_prec = gamma0 / (1.0 + alpha**2)
    prefactor_damp = (alpha * gamma0) / (1.0 + alpha**2)
    
    # Precesión: p = m x Heff
    px = my * Hz - mz * Hy
    py = mz * Hx - mx * Hz
    pz = mx * Hy - my * Hx
    
    # Amortiguamiento: d = m x (m x Heff) = m x p
    dx = my * pz - mz * py
    dy = mz * px - mx * pz
    dz = mx * py - my * px
    
    # Torque total
    dm_dt_x = prefactor_prec * px - prefactor_damp * dx
    dm_dt_y = prefactor_prec * py - prefactor_damp * dy
    dm_dt_z = prefactor_prec * pz - prefactor_damp * dz
    
    return dm_dt_x, dm_dt_y, dm_dt_z




@cuda.jit
def rk4_step_cuda(m, Heff, m_new, dt, gamma0, alpha):
    """
    Kernel de CUDA para avanzar la magnetización un paso dt usando RK4.
    """
    i, j, k = cuda.grid(3)
    Nx, Ny, Nz, _ = m.shape
    
    if i < Nx and j < Ny and k < Nz:
        # Extraer magnetización y campo local a registros ultrarrápidos
        mx = m[i, j, k, 0]
        my = m[i, j, k, 1]
        mz = m[i, j, k, 2]
        
        Hx = Heff[i, j, k, 0]
        Hy = Heff[i, j, k, 1]
        Hz = Heff[i, j, k, 2]
        
        # --- k1 ---
        k1x, k1y, k1z = compute_llg_torque_cuda(mx, my, mz, Hx, Hy, Hz, gamma0, alpha)
        
        # --- k2 ---
        m2x = mx + 0.5 * dt * k1x
        m2y = my + 0.5 * dt * k1y
        m2z = mz + 0.5 * dt * k1z
        k2x, k2y, k2z = compute_llg_torque_cuda(m2x, m2y, m2z, Hx, Hy, Hz, gamma0, alpha)
        
        # --- k3 ---
        m3x = mx + 0.5 * dt * k2x
        m3y = my + 0.5 * dt * k2y
        m3z = mz + 0.5 * dt * k2z
        k3x, k3y, k3z = compute_llg_torque_cuda(m3x, m3y, m3z, Hx, Hy, Hz, gamma0, alpha)
        
        # --- k4 ---
        m4x = mx + dt * k3x
        m4y = my + dt * k3y
        m4z = mz + dt * k3z
        k4x, k4y, k4z = compute_llg_torque_cuda(m4x, m4y, m4z, Hx, Hy, Hz, gamma0, alpha)
        
        # --- Ensamblaje RK4 ---
        m_new_x = mx + (dt / 6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
        m_new_y = my + (dt / 6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y)
        m_new_z = mz + (dt / 6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z)
        
        # Normalización fuerte de la celda
        norm = math.sqrt(m_new_x**2 + m_new_y**2 + m_new_z**2)
        m_new[i, j, k, 0] = m_new_x / norm
        m_new[i, j, k, 1] = m_new_y / norm
        m_new[i, j, k, 2] = m_new_z / norm



@cuda.jit
def heun_step_thermal_cuda(m, Heff, h_th, m_new, dt, gamma0, alpha):
    """
    Kernel de CUDA para el método de Heun (Stratonovich) con ruido térmico pre-generado.
    """
    i, j, k = cuda.grid(3)
    Nx, Ny, Nz, _ = m.shape
    
    if i < Nx and j < Ny and k < Nz:
        
        mx = m[i, j, k, 0]
        my = m[i, j, k, 1]
        mz = m[i, j, k, 2]
        
        # Campo Total = Determinista + Fluctuación Térmica
        Htot_x = Heff[i, j, k, 0] + h_th[i, j, k, 0]
        Htot_y = Heff[i, j, k, 1] + h_th[i, j, k, 1]
        Htot_z = Heff[i, j, k, 2] + h_th[i, j, k, 2]
        
        # --- PREDICTOR (Paso de Euler con el campo total) ---
        dm_dt_Px, dm_dt_Py, dm_dt_Pz = compute_llg_torque_cuda(mx, my, mz, Htot_x, Htot_y, Htot_z, gamma0, alpha)
        
        m_pred_x = mx + dt * dm_dt_Px
        m_pred_y = my + dt * dm_dt_Py
        m_pred_z = mz + dt * dm_dt_Pz
        
        # --- CORRECTOR (Evaluado en m_pred, pero manteniendo el MISMO Htot ruidoso) ---
        dm_dt_Cx, dm_dt_Cy, dm_dt_Cz = compute_llg_torque_cuda(m_pred_x, m_pred_y, m_pred_z, Htot_x, Htot_y, Htot_z, gamma0, alpha)
        
        # --- Promedio Final ---
        m_new_x = mx + 0.5 * dt * (dm_dt_Px + dm_dt_Cx)
        m_new_y = my + 0.5 * dt * (dm_dt_Py + dm_dt_Cy)
        m_new_z = mz + 0.5 * dt * (dm_dt_Pz + dm_dt_Cz)
        
        # Normalización fuerte
        norm = math.sqrt(m_new_x**2 + m_new_y**2 + m_new_z**2)
        m_new[i, j, k, 0] = m_new_x / norm
        m_new[i, j, k, 1] = m_new_y / norm
        m_new[i, j, k, 2] = m_new_z / norm