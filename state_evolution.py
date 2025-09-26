import numpy as np
from scipy.integrate import quad
import scipy.optimize as optimize
import pandas as pd
from numba import njit
from sys import argv


def solve_poly(z, sigma, kappa):
    alpha = 1 / kappa
    R_noise = sigma**2
    a3 = np.sqrt(alpha) * R_noise
    a2 = -(np.sqrt(alpha) * z + R_noise)
    a1 = (z + np.sqrt(alpha) - alpha**(-1 / 2))
    a0 = -1

    coefficients = [a3, a2, a1, a0]

    return np.roots(coefficients)

def edges_rho(sigma, kappa):
    alpha = 1/kappa
    R_noise = sigma**2

    a0 = -12 * R_noise + (4 * R_noise) / alpha + 12 * alpha * R_noise - 4 * alpha**2 * R_noise - 20 * R_noise**2 + R_noise**2 / alpha - 8 * alpha * R_noise**2 - 4 * R_noise**3
    a1 = -(10 * R_noise) / np.sqrt(alpha) + 2 * np.sqrt(alpha) * R_noise + 8 * alpha**(3/2) * R_noise - (2 * R_noise**2) / np.sqrt(alpha) + 8 * np.sqrt(alpha) * R_noise**2
    a2 = 1 - 2 * alpha + alpha**2 + 8 * R_noise - 2 * alpha * R_noise + R_noise**2
    a3 = -2 * np.sqrt(alpha) - 2 * alpha**(3/2) - 2 * np.sqrt(alpha) * R_noise
    a4 = alpha

    coefficients = [a4, a3, a2, a1, a0]

    roots_all = np.roots(coefficients)
    real_roots = np.real(roots_all[np.abs(np.imag(roots_all)) < 1e-10])

    return np.sort(real_roots)

def rho(x, sigma, kappa):        
    return np.max(np.imag(solve_poly(x-1e-12j, sigma, kappa))) / np.pi

def ERM_I_y0(delta, epsilon, kappa, kappa_stud, edges):

    if kappa_stud == 1:
        return epsilon
    
    raise ValueError("kappa_stud must be 1 for the current implementation.")

def ERM_I_precise(delta, epsilon, param_prior):

    Q0, lreg, kappa, kappa_stud, posdef = param_prior

    edges = edges_rho(delta, kappa)
    
    if posdef and 0 < kappa_stud < 1:
        y0 = ERM_I_y0(delta, epsilon, kappa, kappa_stud, edges)
    else:
        y0 = epsilon

    def integrand_plus(x):
        return rho(x, delta, kappa) * (x + epsilon)**2
    
    def integrand_minus(x):
        return rho(x, delta, kappa) * (x - epsilon)**2
    
    val1 = 0
        
    if (not posdef) and len(edges) == 2: 

        if -epsilon < edges[0]:
            val1 = 0
        
        elif -epsilon < edges[1]:
            val1 = quad(integrand_plus, edges[0], -epsilon)[0]
        
        else:
            val1 = quad(integrand_plus, edges[0], edges[1])[0]

    elif (not posdef) and len(edges) == 4: 
    
        if -epsilon < edges[0]:
            val1 = 0

        elif -epsilon < edges[1]:
            val1 = quad(integrand_plus, edges[0], -epsilon)[0]

        elif -epsilon < edges[2]:
            val1 = quad(integrand_plus, edges[0], edges[1])[0]

        elif -epsilon < edges[3]:
            val1 = quad(integrand_plus, edges[0], edges[1])[0] + quad(integrand_plus, edges[2], -epsilon)[0]

        else:
            val1 = quad(integrand_plus, edges[0], edges[1])[0] + quad(integrand_plus, edges[2], edges[3])[0]

    val2 = 0

    if len(edges) == 2: 

        if y0 > edges[1]:
            val2 = 0
        
        elif y0 > edges[0]:
            val2 = quad(integrand_minus, y0, edges[1])[0]
        
        else:
            val2 = quad(integrand_minus, edges[0], edges[1])[0]

    elif len(edges) == 4: 

        if y0 > edges[3]:
            val2 = 0

        elif y0 > edges[2]:
            val2 = quad(integrand_minus, y0, edges[3])[0]

        elif y0 > edges[1]:
            val2 = quad(integrand_minus, edges[2], edges[3])[0]

        elif y0 > edges[0]:
            val2 = quad(integrand_minus, edges[2], edges[3])[0] + quad(integrand_minus, y0, edges[1])[0]

        else:
            val2 = quad(integrand_minus, edges[2], edges[3])[0] + quad(integrand_minus, edges[0], edges[1])[0]

    return val1 + val2

def ERM_partial_I(delta, epsilon, param_prior, variation=1e-8):
    der_delta = (ERM_I_precise(delta + variation/2, epsilon, param_prior) - ERM_I_precise(delta - variation/2, epsilon, param_prior)) / variation
    der_epsilon = (ERM_I_precise(delta, epsilon + variation/2, param_prior) - ERM_I_precise(delta, epsilon - variation/2, param_prior)) / variation
    
    return der_delta, der_epsilon


def ERM_q_eq(q_hat, m_hat, sigma_hat, param_prior, integral, partials):
    Q_0, lreg, kappa, kappa_stud, posdef = param_prior
    return m_hat ** 2 / sigma_hat**2 * integral

def ERM_m_eq(q_hat, m_hat, sigma_hat, param_prior, integral, partials):
    Q_0, lreg, kappa, kappa_stud, posdef = param_prior
    return m_hat / sigma_hat * integral - np.sqrt(q_hat) / 2 / sigma_hat * partials[0] - lreg / sigma_hat * partials[1]

def ERM_sigma_eq(q_hat, m_hat, sigma_hat, param_prior, integral, partials):
    Q_0, lreg, kappa, kappa_stud, posdef = param_prior
    return m_hat / 2 / np.sqrt(q_hat) / sigma_hat * partials[0]


@njit
def _flat_len_to_T(L):
    return int((np.sqrt(8.0 * L + 1.0) - 1.0) / 2.0)

@njit
def sym(A_flat):
    L = len(A_flat)
    T = _flat_len_to_T(L)
    R = np.zeros((T, T), dtype=A_flat.dtype)
    c = 0
    for i in range(T):
        for j in range(i, T):
            v = A_flat[c]
            R[i, j] = v
            R[j, i] = v
            c += 1
    return R

@njit
def desym(A):
    T = A.shape[0]
    L = T * (T + 1) // 2
    out = np.empty(L, dtype=A.dtype)
    c = 0
    for i in range(T):
        for j in range(i, T):
            out[c] = A[i, j]
            c += 1
    return out

@njit
def _rowwise_max(X):
    n, m = X.shape
    out = np.empty(n, dtype=X.dtype)
    for i in range(n):
        row_max = X[i, 0]
        for j in range(1, m):
            if X[i, j] > row_max:
                row_max = X[i, j]
        out[i] = row_max
    return out

@njit
def softmax(x_flat, beta=1.0):
    X = sym(x_flat)

    max_X = _rowwise_max(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = X[i, j] - max_X[i]

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = np.exp(beta * X[i, j])

    for i in range(X.shape[0]):
        s = 0.0
        for j in range(X.shape[1]):
            s += X[i, j]
        invs = 1.0 / s
        for j in range(X.shape[1]):
            X[i, j] *= invs

    return X.ravel()  
@njit
def scale_upper_flat(x_flat):
    """
    Multiplies the symmetric matrix (from x_flat) elementwise by sqrt(1 + I),
    i.e. multiplies diagonal entries by sqrt(2) and off-diagonals by 1,
    then flattens back to upper-triangular.
    """
    L = len(x_flat)
    T = _flat_len_to_T(L)
    X = sym(x_flat)
    rt2 = np.sqrt(2.0)
    for i in range(T):
        for j in range(T):
            if i == j:
                X[i, j] *= rt2

    return desym(X)

@njit
def proximal_argument(sigma, zstar, z, h, additive_noise_in, additive_noise_out,
                      betastar=1.0, beta=1.0):

    zstar_f = zstar + additive_noise_in
    zstar_f = scale_upper_flat(zstar_f)

    h_f = scale_upper_flat(h)

    label1 = softmax(zstar_f, betastar) + additive_noise_out
    label2 = softmax(h_f, beta)

    diff_hz = h - z
    sum1 = 0.0
    for i in range(len(diff_hz)):
        sum1 += diff_hz[i] * diff_hz[i]

    diff_lbl = label1 - label2
    sum2 = 0.0
    for i in range(len(diff_lbl)):
        sum2 += diff_lbl[i] * diff_lbl[i]

    return sum1 / (2.0 * sigma) + sum2

def proximal(sigma, zstar, z, betastar = 1, beta = 1, noise_in=0, noise_out=0):
    init = zstar + np.random.normal(0,1, len(zstar)) * 0.01

    Lout = np.size(softmax(zstar, betastar)) 
    additive_noise_out = np.sqrt(noise_out) * np.random.randn(Lout) 

    additive_noise_in = np.sqrt(noise_in / 2) * np.random.randn(np.size(zstar))

    res = optimize.minimize(lambda x: proximal_argument(sigma, zstar, z, x, additive_noise_in, additive_noise_out, betastar = betastar, beta = beta), init)
    return res.x

def ERM_m_hat_eq(q, m, sigma, param_output, proximal_val, z_all, zstar_all):

    Q_0, T, beta, betastar, noise_in, noise_out = param_output
    
    samples = proximal_val.shape[0]

    res = 0
    for i in range(0,samples):
        prox = proximal_val[i]
        z = z_all[i]
        zstar = zstar_all[i]
        res += np.sum((prox - z) * (- m * z + q * zstar)) / sigma / (Q_0 * q - m**2)
    return res / samples


def ERM_q_hat_eq(q, m, sigma, param_output, proximal_val, z_all, zstar_all):

    Q_0, T, beta, betastar, noise_in, noise_out = param_output

    samples = proximal_val.shape[0]

    res = 0

    for i in range(0,samples):
        prox = proximal_val[i]
        z = z_all[i]
        zstar = zstar_all[i]
        res += np.sum((prox - z)**2) / sigma**2

    return res / samples

def ERM_sigma_hat_eq(q, m, sigma, param_output, proximal_val, z_all, zstar_all):

    Q_0, T, beta, betastar, noise_in, noise_out = param_output
    
    samples = proximal_val.shape[0]

    res = 0

    for i in range(0,samples):
        prox = proximal_val[i]
        z = z_all[i]
        zstar = zstar_all[i]
        res += np.sum((prox - z) * (Q_0 * z - m * zstar)) / sigma / (Q_0 * q - m**2)

    return res / samples



def ERM_state_evolution_equations(overlaps, alpha, param_prior, param_output, samples=1000):
    q, m, sigma, q_hat, m_hat, sigma_hat = overlaps

    lreg = param_prior[1]

    integral = ERM_I_precise(np.sqrt(q_hat)/m_hat, 2 * lreg /m_hat, param_prior)
    partials = ERM_partial_I(np.sqrt(q_hat)/m_hat, 2 * lreg /m_hat, param_prior)

    L = int(T * (T+1) / 2)
    zstar_all = np.zeros((samples, L))
    z_all = np.zeros((samples, L))
    proximal_val_all = np.zeros((samples, L))
    for i in range(0,samples):
        zstar, z = np.random.multivariate_normal([0,0],[[Q_0,m],[m,q]], L).T   
        proximal_val = proximal(sigma, zstar, z, betastar = betastar, beta = beta, noise_in=noise_in, noise_out=noise_out) 
        zstar_all[i] = zstar
        z_all[i] = z
        proximal_val_all[i] = proximal_val

    q_new = ERM_q_eq(q_hat, m_hat, sigma_hat, param_prior, integral, partials)
    m_new = ERM_m_eq(q_hat, m_hat, sigma_hat, param_prior, integral, partials)
    sigma_new = ERM_sigma_eq(q_hat, m_hat, sigma_hat, param_prior, integral, partials)
    q_hat_new = 2 * alpha * ERM_q_hat_eq(q, m, sigma, param_output, proximal_val_all, z_all, zstar_all)
    m_hat_new = 2 * alpha *  ERM_m_hat_eq(q, m, sigma, param_output, proximal_val_all, z_all, zstar_all)
    sigma_hat_new = - 2 * alpha *  ERM_sigma_hat_eq(q, m, sigma, param_output, proximal_val_all, z_all, zstar_all)

    return np.array([q_new, m_new, sigma_new, q_hat_new, m_hat_new, sigma_hat_new])



def ERM_solution(alpha, Q_0, lreg, kappa, kappa_stud, noise_in, noise_out, posdef, T, beta, betastar, q = 0.3, m = 0.2, sigma = 0.1, damping=1.0, max_iter=50000, toll=1e-5, samples=1000):

    param_prior = [Q_0, lreg, kappa, kappa_stud, posdef]
    param_output = [Q_0, T, beta, betastar, noise_in, noise_out]


    L = int(T * (T+1) / 2)
    zstar_all = np.zeros((samples, L))
    z_all = np.zeros((samples, L))
    proximal_val_all = np.zeros((samples, L))
    for i in range(0,samples):
        zstar, z = np.random.multivariate_normal([0,0],[[Q_0,m],[m,q]], L).T   
        proximal_val = proximal(sigma, zstar, z, betastar = betastar, beta = beta, noise_in=noise_in, noise_out=noise_out) 
        zstar_all[i] = zstar
        z_all[i] = z
        proximal_val_all[i] = proximal_val

    q_hat =  2 * alpha * ERM_q_hat_eq(q, m, sigma, param_output, proximal_val_all, z_all, zstar_all)
    m_hat =  2 * alpha * ERM_m_hat_eq(q, m, sigma, param_output, proximal_val_all, z_all, zstar_all)
    sigma_hat = -  2 * alpha * ERM_sigma_hat_eq(q, m, sigma, param_output, proximal_val_all, z_all, zstar_all)

    overlaps = np.array([q, m, sigma, q_hat, m_hat, sigma_hat])

    epsilon = 2 / overlaps[4]
    delta = np.sqrt(overlaps[3]) / overlaps[4]

    _, D01 = ERM_partial_I(delta, epsilon*lreg, param_prior)
    loss = delta**2/4/epsilon**2 - lreg/2 * D01

    df = pd.DataFrame({
        "q": [q],
        "m": [m],
        "sigma": [sigma],
        "q_hat": [q_hat],
        "m_hat": [m_hat],
        "sigma_hat": [sigma_hat],
        "MSE": [Q_0 + q - 2 * m],
        "loss": [loss]
    })
    df.to_csv(f"data/alpha_{alpha}_kappa_{kappa}_lreg_{lreg}_noiseIN_{noise_in}_noiseOUT_{noise_out}_T_{T}.csv", index=False)


    for i in range(max_iter):
        new_overlaps = ERM_state_evolution_equations(overlaps, alpha, param_prior, param_output, samples=samples)

        err_toll = np.linalg.norm(new_overlaps - overlaps)

        epsilon = 2 / overlaps[4]
        delta = np.sqrt(overlaps[3]) / overlaps[4]

        _, D01 = ERM_partial_I(delta, epsilon*lreg, param_prior)
        loss = delta**2/4/epsilon**2 - lreg/2 * D01


        df_new = pd.DataFrame({
            "q": [overlaps[0]],
            "m": [overlaps[1]],
            "sigma": [overlaps[2]],
            "q_hat": [overlaps[3]],
            "m_hat": [overlaps[4]],
            "sigma_hat": [overlaps[5]],
            "MSE": [Q_0 + overlaps[0] - 2 * overlaps[1]],
            "loss": [loss]
        })
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(f"data/alpha_{alpha}_kappa_{kappa}_lreg_{lreg}_noiseIN_{noise_in}_noiseOUT_{noise_out}_T_{T}.csv", index=False)

        if err_toll < toll:
            return overlaps, Q_0 + new_overlaps[0] - 2 * new_overlaps[1], loss
        

        overlaps = (1-damping) * overlaps + damping * new_overlaps

    return overlaps, float('NaN'), float('NaN')


if __name__ == "__main__":

    alpha = int(argv[1])
    lreg = float(argv[2]) 
    kappa = float(argv[3])
    noise_in = float(argv[4])
    noise_out = float(argv[5])

    Q_0 = 1 + kappa
    beta = 1
    betastar = 1


    T = 2
    kappa_stud = 1
    posdef = True
    ERM_solution(alpha, Q_0, lreg, kappa, kappa_stud, noise_in, noise_out, posdef, T, beta, betastar, q = 0.6, m = 0.2, sigma = 1., damping=0.8, max_iter=1000, toll=1e-5, samples=int(1e3))