import numpy as np
import datetime
import time
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .huawei import *
from .NBI import *
from .setup import *



def latin_hypercube(n_samples, dim, bounds):
    """
    Gera amostras usando Latin Hypercube Sampling.
    Dentro dos intervalos: np.random.uniform escolhe um ponto aleatório em cada subintervalo.

    Ordem dos intervalos: np.random.shuffle embaralha a ordem dos intervalos.

    Resultado: Pontos distribuídos uniformemente, mas com variação entre execuções (a menos que você fixe a seed).
    Parâmetros:
        n_samples (int): Número de amostras.
        dim (int): Dimensão do espaço (2 para tilt/azimute).
        bounds (list): Lista de tuplas com limites [(min, max), ...].
    
    Retorna:
        np.array: Amostras no espaço normalizado [0, 1]^dim.

    """
    samples = np.zeros((n_samples, dim))
    for i in range(dim):
        # Divide o espaço em n_samples intervalos
        intervals = np.linspace(0, 1, n_samples + 1)
        # Amostra aleatoriamente dentro de cada intervalo
        points = np.random.uniform(intervals[:-1], intervals[1:], size=n_samples)
        # Embaralha para garantir aleatoriedade
        np.random.shuffle(points)
        samples[:, i] = points
    return samples

def maximize_users(connection,
                    user_function,
                    cell, 
                    local_cell_id,
                    sector_split_group_id,
                    tilt_atual,
                    azimute_atual,
                    n_iter = TRIALS, init_points=6, noise_level=1e-2,
                    stop_confidence = CONFIDENCE 
                    ): #1e-6
    """Bayesian optimization with Gaussian Process and Expected Improvement."""
    t_bounds = (TILT_RANGE[0], TILT_RANGE[1]) #-2 para geral e -15 para estádios
    a_bounds = (AZIMUTH_RANGE[0], AZIMUTH_RANGE[1])
    bounds = np.array([t_bounds, a_bounds])
    
    # Initial points based on predetermined positions
#    initial_positions = [(2, -15), (2, 15), (10, 0) ]
    initial_positions = [(tilt_atual, azimute_atual),(6,0), (1,15), (10,15), (10,-15), (1,-15)]
    
    X = np.array(initial_positions[:init_points])
    y = np.array([user_function(connection, cell, t, a, local_cell_id, sector_split_group_id) for t, a in X])
    evaluated_points = set(map(tuple, X))
    
    # Kernel with optimized hyperparameters and noise level (alpha)
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_level,  # Initial noise level
        n_restarts_optimizer=50,  # Optimize kernel hyperparameters and alpha
        normalize_y=True
    )
    gp.fit(X, y)
    
    def expected_improvement(x):
        """Vectorized Expected Improvement."""
        x = np.atleast_2d(x)
        mu, sigma = gp.predict(x, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        y_max = np.max(y)
        improvement = mu - y_max
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei  # Negative for minimization
    
    # Bayesian optimization loop
    for iteracoes in range(n_iter): # até 30 chamadas a função de usuário (acesso ao HUAWEI e aguarda x Segundos)
        # Find candidate with the highest EI using gradient-based optimization
        best_x = None
        best_ei = np.inf  # We minimize -EI
        # Geração de pontos iniciais via Latin Hypercube
        lhs_samples = latin_hypercube(n_samples=250, dim=2, bounds=[t_bounds, a_bounds])
        
        # Normalização para os limites reais
        t_scaled = lhs_samples[:, 0] * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        a_scaled = lhs_samples[:, 1] * (a_bounds[1] - a_bounds[0]) + a_bounds[0]
        
        # Conversão para inteiros (mantendo precisão para otimização contínua)
        t_ints = np.round(t_scaled).astype(int)
        a_ints = np.round(a_scaled).astype(int)
        
        # Garantia de limites físicos
        t_ints = np.clip(t_ints, t_bounds[0], t_bounds[1])
        a_ints = np.clip(a_ints, a_bounds[0], a_bounds[1])

        # Multi-start optimization to avoid local minima
        # calcula-se o valor da funçao estimada em todos os 50 pontos e escolhe-se o melhor
        # o melhor é o que tiver maior Expected Improvement
        # o maior EI é o que tem maior incerteza, que é o que está na borda do intervalo de confianca frequentista
        # vide figura 1 de A Tutorial on Bayesian Optimization.
        for i in range(250):  # 5
            x0 = np.array([t_ints[i], a_ints[i]], dtype=np.float64)
            res = minimize(expected_improvement, x0, bounds=bounds, method='L-BFGS-B')
            
            if res.fun < best_ei and not any(np.allclose(res.x, x, atol=1e-3) for x in evaluated_points):
                best_ei = res.fun
                best_x = res.x
        
        if best_x is None:
            continue  # Fallback to random sampling if all candidates are duplicates
        
            # Early stopping check: Probability of Improvement
        mu, sigma = gp.predict(best_x.reshape(1, -1), return_std=True)
        y_max = np.max(y)
        z = (y_max - mu) / sigma
        prob_improvement = 1 - norm.cdf(z)
        if prob_improvement < (1 - stop_confidence):
            break  # Para se o incremento esperado for menor que 5%
    
        # Arredondamento final para posições físicas
        best_x = np.round(best_x, decimals=1)
        # Evaluate and update

        new_y = user_function(connection, cell, best_x[0], round(best_x[1]), local_cell_id, sector_split_group_id)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Cell: {cell}; Tilt: {best_x[0]}; Azimuth: {round(best_x[1])};  Usuarios: {new_y}; Iterations: {iteracoes + 1}; Confidence: {(1 - prob_improvement)}\n"
        #print(f"[{timestamp}] Cell: {cell}; Tilt: {(best_x[0])}; Azimuth: {(best_x[1])};  Iterations: {iteracoes + 1}; Confidence: {(1 - prob_improvement)}")
        with open(LOG_RESULTS, "a") as f:
             f.write(log_entry)
        X = np.vstack([X, best_x])
        y = np.append(y, new_y)
        evaluated_points.add(tuple(best_x))
        gp.fit(X, y)  # Re-fit GP with updated data and optimized alpha
    
    idx_best = np.argmax(y)
    # Before logging
    prob_improvement = float(prob_improvement)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]  Melhor posicao ; Cell: {cell}; Tilt: {(X[idx_best, 0])}; Azimuth: {int(X[idx_best, 1])};  Iterations: {iteracoes + 1}; Confidence: {(1 - prob_improvement):.4f}\n"
    with open(LOG_RESULTS, "a") as f:
        f.write(log_entry)
    return ((X[idx_best, 0]), round(X[idx_best, 1])), y[idx_best]


def AjustForBestPos(connection,
                    cell,
                    tilt,
                    azimuth,
                    best_val,
                    local_cell_id,
                    sector_split_group_id ) -> None:

    signal = 1 if tilt >= 0 else -1
    tilt_int = int(tilt)  # get integer part of tilt
    tilt_frac = signal * round((tilt - tilt_int) * 10)  # get fractional part (tenths)

    connection.send_command(f"MOD SECTORSPLITCELL:LOCALCELLID={local_cell_id},SECTORSPLITGROUPID={sector_split_group_id},CELLBEAMTILT={tilt_int},CELLBEAMTILTFRACTIONPART={tilt_frac},CELLBEAMAZIMUTHOFFSET={azimuth};")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Melhor posição {cell}: TILT={tilt}, AZIMUTE={azimuth} com {best_val} usuários\n"
    print(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)



def otimizar_site(self, conn, site):
    """
    Optimizes network configuration for a given site by adjusting parameters like cell beam tilt,
    azimuth offset, and power levels for primary and secondary cells. Saves results to an Excel report.
    Args:
        Mae (object): Provides methods for network operations.
        site (str): Site identifier.
    Notes:
        - Uses helper functions for parameter adjustments.
        - Saves report to 'reports/CE05CW.xlsx'.
    """
    conn.select_ne(site)
    conn.get_network_status()
    cells = conn.df_global['Cell Name'].unique()
    try:
        for cell in (cells):
            if primary_cell(cell):
                local_cell_id = conn.df_global.loc[conn.df_global['Cell Name'] == cell, 'Local Cell ID'].values[0]
                sector_split_group_id = conn.df_global.loc[conn.df_global['Cell Name'] == cell, 'Sector Split Group ID'].values[0]                        
                tilt_atual = conn.df_global[conn.df_global['Cell Name'] == cell]['Cell Beam Tilt(degree)'].values[0]
                azimute_atual = conn.df_global[conn.df_global['Cell Name'] == cell]['Cell Beam Azimuth Offset(degree)'].values[0]
                usuarios = conn.df_global[conn.df_global['Cell Name'] == cell]['Cell Total Counter'].values[0]
                if usuarios < MaxUsers(cell):
                    AjusteDePotencia(conn, 'aumentar', cell, local_cell_id)
                    best_pos, best_val = maximize_users(conn,
                                                        FunctionUsers,
                                                        cell,
                                                        local_cell_id,
                                                        sector_split_group_id,
                                                        tilt_atual,
                                                        azimute_atual
                                                        )
                    
                    AjustForBestPos(conn,
                                    cell,
                                    best_pos[0],
                                    best_pos[1],
                                    best_val,
                                    local_cell_id,
                                    sector_split_group_id)
                    
                    try:  
                        SecondaryCell = secondary_cell(cell)
                        try:
                            local_cell_id = conn.df_global.loc[conn.df_global['Cell Name'] == SecondaryCell, 'Local Cell ID'].values[0]
                            sector_split_group_id = conn.df_global.loc[conn.df_global['Cell Name'] == SecondaryCell, 'Sector Split Group ID'].values[0]

                        except:
                            if SecondaryCell.startswith('4C-'):
                                SecondaryCell = '4G-' + SecondaryCell[3:]
                                local_cell_id = conn.df_global.loc[conn.df_global['Cell Name'] == SecondaryCell, 'Local Cell ID'].values[0]
                                sector_split_group_id = conn.df_global.loc[conn.df_global['Cell Name'] == SecondaryCell, 'Sector Split Group ID'].values[0]
                            else:
                                pass
                            
                        AjustForBestPos(conn,
                                    SecondaryCell,
                                    best_pos[0],
                                    best_pos[1],
                                    best_val,
                                    local_cell_id,
                                    sector_split_group_id)
                    except Exception as e:
                        print(f"Erro: {str(e)}")
                        continue
                    
                else:
                    AjusteDePotencia(conn, 'reduzir',cell, local_cell_id)
                            
            else:
                continue
            self.get_network_status()
            df_report = pd.concat([df_report, conn.df_global], ignore_index=True)
            site_date_time = f"{site}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            df_report.to_excel(f'reports/{site_date_time}.xlsx', index=False)
    except Exception as e:
        print(f"Erro: {str(e)}")
        
    
