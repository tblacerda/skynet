import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize


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
        return -ei
    
    steps_used = 0
    for steps_used in range(n_iter):
        best_x = None
        best_ei = np.inf
        
        # Candidate selection
        for _ in range(50):
            x0 = np.array([
                np.random.randint(t_bounds[0], t_bounds[1] + 1),
                np.random.randint(a_bounds[0], a_bounds[1] + 1)
            ])
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
        
        # Update model
        best_x = best_x.round().astype(int)
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
    
    for round_num in range(num_rounds):
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
    results_df.to_excel('optimization_benchmark_90_nu2.5.xlsx', index=False)
    print(results_df.describe())

    # Print summary statistics
    print("\nOptimization Performance Summary:")
    print(f"Average steps taken: {results_df['Steps_Taken'].mean():.1f}")
    print(f"Early stopping rate: {(results_df['Steps_Taken'] < 15).mean() * 100:.1f}%")