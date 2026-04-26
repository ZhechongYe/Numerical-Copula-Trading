import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, List

# ---------------------------------------------------------
# 1. Pseudo-observations (ECDF)
# ---------------------------------------------------------

def get_pseudo_observations(spread: np.ndarray) -> np.ndarray:
    """
    Convert spread series to uniform distribution [0, 1] using ECDF.
    """
    n = len(spread)
    ranks = stats.rankdata(spread, method='average')
    return ranks / (n + 1)

# ---------------------------------------------------------
# 2. Log-Likelihood Functions
# ---------------------------------------------------------

def gaussian_log_likelihood(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    rho = theta[0] if isinstance(theta, (list, np.ndarray)) else theta
    if abs(rho) >= 1.0: return 1e10
    
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    # Copula Density: c(u,v) = 1/sqrt(1-rho^2) * exp(-(rho^2*(x^2+y^2) - 2*rho*x*y) / (2*(1-rho^2)))
    term1 = -0.5 * np.log(1 - rho**2)
    term2 = -(rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))
    return -np.sum(term1 + term2)

def t_copula_log_likelihood(params: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    rho, nu = params[0], params[1]
    if abs(rho) >= 1.0 or nu <= 2.0: return 1e10
    
    x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    # Log-density of t-copula involves Gamma functions and the determinant of correlation matrix
    # Using the standard density formula for 2D Student-t copula
    part1 = stats.t.logpdf(x, nu) + stats.t.logpdf(y, nu)
    
    # Bivariate t-density part
    rho_mat = np.array([[1, rho], [rho, 1]])
    inv_rho = np.linalg.inv(rho_mat)
    val_vec = np.column_stack([x, y])
    quad_form = np.sum((val_vec @ inv_rho) * val_vec, axis=1)
    
    log_c_nom = stats.gamma.loggamma((nu + 2)/2) + stats.gamma.loggamma(nu/2)
    log_c_den = 2 * stats.gamma.loggamma((nu + 1)/2) + 0.5 * np.log(1 - rho**2)
    part2 = log_c_nom - log_c_den - ((nu + 2)/2) * np.log(1 + quad_form/nu)
    
    return -np.sum(part2 - part1)

def clayton_log_likelihood(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    if theta <= 0: return 1e10
    term = (1 + theta) * (u * v)**(-1 - theta) * (u**-theta + v**-theta - 1)**(-2 - 1/theta)
    return -np.sum(np.log(np.maximum(term, 1e-12)))

def gumbel_log_likelihood(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    if theta <= 1: return 1e10
    ln_u, ln_v = -np.log(u), -np.log(v)
    a = ln_u**theta + ln_v**theta
    term = np.exp(-a**(1/theta)) * a**(2/theta - 2) * (ln_u * ln_v)**(theta - 1) * (1 + (theta - 1) * a**(-1/theta))
    return -np.sum(np.log(np.maximum(term, 1e-12)))

def frank_log_likelihood(theta: float, u: np.ndarray, v: np.ndarray) -> float:
    if abs(theta) < 1e-5: return 1e10
    num = theta * (1 - np.exp(-theta)) * np.exp(-theta * (u + v))
    den = ((1 - np.exp(-theta)) - (1 - np.exp(-theta*u)) * (1 - np.exp(-theta*v)))**2
    return -np.sum(np.log(np.maximum(num / den, 1e-12)))

# ---------------------------------------------------------
# 3. Main Fitting Engine with Multi-Method Optimization
# ---------------------------------------------------------

def fit_best_copula(u: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
    results = []
    methods = ['L-BFGS-B', 'Nelder-Mead', 'BFGS']
    
    # Format: (name, func, initial_guess, bounds, k_params)
    families = [
        ('clayton', clayton_log_likelihood, [1.0], [(0.001, 20.0)], 1),
        ('gumbel', gumbel_log_likelihood, [2.0], [(1.001, 20.0)], 1),
        ('frank', frank_log_likelihood, [1.0], [(-20.0, 20.0)], 1),
        ('gaussian', gaussian_log_likelihood, [0.5], [(-0.99, 0.99)], 1),
        ('t', t_copula_log_likelihood, [0.5, 4.0], [(-0.99, 0.99), (2.01, 50.0)], 2)
    ]
    
    for name, loglik_func, guess, bnds, k in families:
        best_res = None
        for method in methods:
            try:
                res = minimize(loglik_func, x0=guess, args=(u, v), 
                               bounds=bnds if method == 'L-BFGS-B' else None, 
                               method=method)
                if res.success:
                    best_res = res
                    break
            except: continue
        
        if best_res:
            aic = 2 * k + 2 * best_res.fun
            results.append({'type': name, 'theta': best_res.x, 'aic': aic})

    return min(results, key=lambda x: x['aic']) if results else {'type': 'none', 'aic': np.inf}
    #return results

# ---------------------------------------------------------
# 4. H-Index Calculation
# ---------------------------------------------------------

def calculate_h_index(u: float, v: float, theta: Any, copula_type: str, direction: str) -> float:
    if direction == '2|1': u, v = v, u
    u, v = np.clip(u, 1e-7, 1-1e-7), np.clip(v, 1e-7, 1-1e-7)

    if copula_type == 'clayton':
        t = theta[0]
        return v**(-t-1) * (u**-t + v**-t - 1)**(-1/t - 1)
    elif copula_type == 'gumbel':
        t = theta[0]
        ln_u, ln_v = -np.log(u), -np.log(v)
        return np.exp(-(ln_u**t + ln_v**t)**(1/t)) * (ln_u**t + ln_v**t)**(1/t - 1) * (ln_v**(t-1)) / v
    elif copula_type == 'frank':
        t = theta[0]
        return (np.exp(-t*v) * (np.exp(-t*u)-1)) / ((np.exp(-t)-1) + (np.exp(-t*u)-1)*(np.exp(-t*v)-1))
    elif copula_type == 'gaussian':
        rho = theta[0]
        return stats.norm.cdf((stats.norm.ppf(u) - rho * stats.norm.ppf(v)) / np.sqrt(1 - rho**2))
    elif copula_type == 't':
        rho, nu = theta[0], theta[1]
        x_u, x_v = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        return stats.t.cdf((x_u - rho * x_v) * np.sqrt((nu + 1) / (nu + x_v**2)) / np.sqrt(1 - rho**2), nu + 1)
    return 0.5