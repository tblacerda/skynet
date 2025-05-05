import telnetlib
import re
import pandas as pd
import schedule
import time
import datetime
import functools
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
VELOCIDADE = 1 # segundos. Tempo de espera entre comandos

# Global variable to store the result
dfGlobal = None #informações consolidadas

def save_df_global() -> None:
    """Save dfGlobal to disk"""
    global dfGlobal
    if dfGlobal is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dfGlobal.to_excel(f'network_status_{timestamp}.csv', index=False)
# Decorator to log Telnet commands


def maximize_users(user_function, tilt_atual, azimute_atual, n_iter=10, init_points=1, noise_level=1e-2, stop_confidence=0.95):
    """Bayesian optimization with Gaussian Process and Expected Improvement, including early stopping."""
    t_bounds = (-2, 15)
    a_bounds = (-30, 30)
    bounds = np.array([t_bounds, a_bounds])
    
    #initial_positions = [(15, 30), (6, -15), (6, 15)]
    initial_positions = [(tilt_atual, azimute_atual)]
    X = np.array(initial_positions[:init_points])
    y = np.array([user_function(t, a) for t, a in X])
    evaluated_points = set(map(tuple, X))
    
    kernel = Matern(nu=2.5)
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
        return -ei  # Minimize negative EI
    
    for _ in range(n_iter):
        best_x = None
        best_ei = np.inf
        
        # Multi-start optimization to find the best candidate
        for _ in range(50):
            x0 = np.array([
                np.random.randint(t_bounds[0]//1, t_bounds[1]//1 + 1) * 1,
                np.random.randint(a_bounds[0]//1, a_bounds[1]//1 + 1) * 1
            ])
            res = minimize(expected_improvement, x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_ei and not any(np.allclose(res.x, x, atol=1e-3) for x in evaluated_points):
                best_ei = res.fun
                best_x = res.x
        
        if best_x is None:
            continue
        
        # Early stopping check: Probability of Improvement
        mu, sigma = gp.predict(best_x.reshape(1, -1), return_std=True)
        y_max = np.max(y)
        z = (y_max - mu) / sigma
        prob_improvement = 1 - norm.cdf(z)
        if prob_improvement < (1 - stop_confidence):
            break  # Stop if confidence threshold met
        
        # Round to nearest integer
        best_x[0] = int(round(best_x[0]))
        best_x[1] = int(round(best_x[1]))
        
        # Evaluate and update model
        new_y = user_function(int(best_x[0]), int(best_x[1]))
        X = np.vstack([X, best_x])
        y = np.append(y, new_y)
        evaluated_points.add(tuple(best_x))
        gp.fit(X, y)
    
    idx_best = np.argmax(y)
    return (int(X[idx_best, 0]), int(X[idx_best, 1])), y[idx_best]

# Função hipotética que retorna o número de usuários para uma posição
def FunctionUsers(tilt, azimuth):
   
    print(f"Tilt: {tilt}, Azimuth: {azimuth}, Users: {data[tilt+2, azimuth+30]}")
    return  data[tilt+2, azimuth+30]


def main():
    global data
    df = pd.read_excel('dadosSinteticos.xlsx', header=None)
    data = df.to_numpy()



    best_pos, best_val = maximize_users(FunctionUsers, n_iter=20)
    print(f"Best Position: {best_pos}, Best Value: {best_val}")

if __name__ == "__main__":
    main()

