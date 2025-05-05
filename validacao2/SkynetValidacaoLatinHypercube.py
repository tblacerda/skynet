import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm

def latin_hypercube(n_samples, dim, bounds):
    """
    Gera amostras usando Latin Hypercube Sampling.
    
    Parâmetros:
        n_samples (int): Número de amostras.
        dim (int): Dimensão do espaço (2 para tilt/azimute).
        bounds (list): Lista de tuplas com limites [(min, max), ...].
    
    Retorna:
        np.array: Amostras no espaço normalizado [0, 1]^dim.
        Dentro dos intervalos: np.random.uniform escolhe um ponto aleatório em cada subintervalo.

    Ordem dos intervalos: np.random.shuffle embaralha a ordem dos intervalos.

    Resultado: Pontos distribuídos uniformemente, 
    mas com variação entre execuções (a menos que você fixe a seed).
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


def maximize_users(user_function, n_iter=10, init_points=1, noise_level=1e-2, stop_confidence=0.90):
    """Bayesian optimization with random initial points and early stopping tracking."""
    t_bounds = (-2, 15)
    a_bounds = (-30, 30)
    bounds = np.array([t_bounds, a_bounds])
    
    # Generate random initial points
    X = []
    evaluated_points = set()
    while len(X) < init_points:
        t = np.random.randint(t_bounds[0], t_bounds[1] + 1)
        a = np.random.randint(a_bounds[0], a_bounds[1] + 1)
        if (t, a) not in evaluated_points:
            X.append([t, a])
            evaluated_points.add((t, a))
    
    X = np.array(X)
    y = np.array([user_function(t, a) for t, a in X])
    
    kernel = Matern(nu=1.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_level,
        n_restarts_optimizer=50,
        normalize_y=True
    )
    gp.fit(X, y)
    
    def expected_improvement(x):
        x = np.atleast_2d(x)
        mu, sigma = gp.predict(x, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        y_max = np.max(y)
        improvement = mu - y_max
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei
    
    steps_used = 0
    for steps_used in range(n_iter):
        best_x = None
        best_ei = np.inf
        
        # Geração de pontos iniciais via Latin Hypercube
        lhs_samples = latin_hypercube(n_samples=50, dim=2, bounds=[t_bounds, a_bounds])
        
        # Normalização para os limites reais
        t_scaled = lhs_samples[:, 0] * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        a_scaled = lhs_samples[:, 1] * (a_bounds[1] - a_bounds[0]) + a_bounds[0]
        
        # Conversão para inteiros (mantendo precisão para otimização contínua)
        t_ints = np.round(t_scaled).astype(int)
        a_ints = np.round(a_scaled).astype(int)
        
        # Garantia de limites físicos
        t_ints = np.clip(t_ints, t_bounds[0], t_bounds[1])
        a_ints = np.clip(a_ints, a_bounds[0], a_bounds[1])

        # Candidate selection
        for i in range(50):
            x0 = np.array([t_ints[i], a_ints[i]], dtype=np.float64)
            res = minimize(expected_improvement, x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_ei and not any(np.allclose(res.x, x, atol=1e-3) for x in evaluated_points):
                best_ei = res.fun
                best_x = res.x
        
        if best_x is None:
            continue
        
        # Early stopping check
        mu, sigma = gp.predict(best_x.reshape(1, -1), return_std=True)
        y_max = np.max(y)
        z = (y_max - mu) / sigma
        prob_improvement = 1 - norm.cdf(z)
        if prob_improvement < (1 - stop_confidence):
            break
        
        # Arredondamento final para posições físicas
        best_x = np.round(best_x).astype(int)
        new_y = user_function(*best_x)
        X = np.vstack([X, best_x])
        y = np.append(y, new_y)
        evaluated_points.add(tuple(best_x))
        gp.fit(X, y)
    
    idx_best = np.argmax(y)
    return (X[idx_best, 0], X[idx_best, 1]), y[idx_best], steps_used + 1

def run_optimization_benchmark(data, num_rounds=100, max_iter=15, init_points=3):
    """Run multiple optimization rounds and collect statistics."""
    results = []
    
    def simulator(tilt, azimuth):
        return data[tilt + 2, azimuth + 30]
    
    for round_num in tqdm(range(num_rounds), desc="Optimization rounds"):
        (best_t, best_a), best_val, steps = maximize_users(
            simulator,
            n_iter=max_iter,
            init_points=init_points
        )
        
        results.append({
            'Round': round_num + 1,
            'Best_Tilt': best_t,
            'Best_Azimuth': best_a,
            'Best_Value': best_val,
            'Steps_Taken': steps,
            'Max_Iterations': max_iter,
            'Initial_Points': init_points
        })
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Load your synthetic data
    df = pd.read_excel('dadosSinteticos.xlsx', header=None)
    data = df.to_numpy()
    
    # Run benchmark
    results_df = run_optimization_benchmark(
        data=data,
        num_rounds=1000,
        max_iter=50,
        init_points=1,    
    )
    
    # Save and display results
    results_df.to_excel('optimization_benchmark_90_nu1.5_LatingHyperCube500.xlsx', index=False)
    print(results_df.describe())

    # Print summary statistics
    print("\nOptimization Performance Summary:")
    print(f"Average steps taken: {results_df['Steps_Taken'].mean():.1f}")
    print(f"Early stopping rate: {(results_df['Steps_Taken'] < 15).mean() * 100:.1f}%")