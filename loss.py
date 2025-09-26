import numpy as np
from scipy.integrate import quad
import scipy.optimize as optimize
from scipy.optimize import root
import pandas as pd
from numba import njit


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
def sym(A):
    T = int((np.sqrt(8*len(A) + 1) - 1)/2)
    c = 0
    R = np.zeros((T,T))
    for i in range(T):
        for j in range(i,T):
            R[i,j]=A[c]
            R[j,i]=A[c]
            c = c+1
    return R

def desym(A):
    return A[np.triu_indices_from(A)]

@njit
def softmax(x, beta=1):
     

    X = sym(x)
     
     
     
     
    max_X = np.array([np.max(row) for row in X])
    X = X - max_X.reshape(-1,1)
    P = np.exp(beta*X)    
    res = (P.T / np.sum(P,1)).T
    return res.flatten()

 
 
def proximal_argument(sigma, zstar, z, h, additive_noise_in, additive_noise_out, betastar = 1, beta = 1):    
     

     
    zstar_f = zstar + additive_noise_in   
    zstar_f = sym(zstar_f)  
    zstar_f = zstar_f * np.sqrt(np.ones_like(zstar_f) + np.identity(np.shape(zstar_f)[1]))  
    zstar_f = desym(zstar_f)  

     
    h_f = sym(h)  
    h_f = h_f * np.sqrt(np.ones_like(h_f) + np.identity(np.shape(h_f)[1]))  
    h_f = desym(h_f)  

     
    label1 = softmax(zstar_f, betastar) + additive_noise_out
    label2 = softmax(h_f, beta)

     
    sum1 = np.dot(h - z, h - z)
    sum2 = np.dot(label1 - label2, label1 - label2)

    return sum1/(2 * sigma) + sum2

def proximal(sigma, zstar, z, betastar = 1, beta = 1, noise_in=0, noise_out=0):
     
    init = zstar + np.random.normal(0,1, len(zstar)) * 0.01

    Lout = np.size(softmax(zstar, betastar))  
    additive_noise_out = np.sqrt(noise_out) * np.random.randn(Lout)  

    additive_noise_in = np.sqrt(noise_in / 2) * np.random.randn(np.size(zstar))

    res = optimize.minimize(lambda x: proximal_argument(sigma, zstar, z, x, additive_noise_in, additive_noise_out, betastar = betastar, beta = beta), init)
    return res.x, res.fun

 

import numpy as np
from numba import njit
from scipy import optimize

 

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
def _scale_diag_sqrt2_inplace(X):
     
    rt2 = np.sqrt(2.0)
    T = X.shape[0]
    for i in range(T):
        X[i, i] *= rt2

@njit
def _scale_upper_flat(x_flat):
     
    X = sym(x_flat)
    _scale_diag_sqrt2_inplace(X)
    return desym(X)

@njit
def softmax(x_flat, beta=1.0):
     
    X = sym(x_flat)

     
    max_X = _rowwise_max(X)
    T = X.shape[0]
    for i in range(T):
        row_max = max_X[i]
        for j in range(T):
            X[i, j] = X[i, j] - row_max

     
    for i in range(T):
        for j in range(T):
            X[i, j] = np.exp(beta * X[i, j])
    for i in range(T):
        s = 0.0
        for j in range(T):
            s += X[i, j]
        invs = 1.0 / s
        for j in range(T):
            X[i, j] *= invs

    return X.ravel()


@njit
def proximal_argument(sigma, zstar, z, h, additive_noise_in, additive_noise_out,
                      betastar=1.0, beta=1.0):
     
    zstar_f = zstar + additive_noise_in
    zstar_f = _scale_upper_flat(zstar_f)

     
    h_f = _scale_upper_flat(h)

     
    label1 = softmax(zstar_f, betastar) + additive_noise_out
    label2 = softmax(h_f, beta)

     
    sum1 = 0.0
    for i in range(len(h)):
        d = h[i] - z[i]
        sum1 += d * d

    sum2 = 0.0
    for i in range(len(label1)):
        d = label1[i] - label2[i]
        sum2 += d * d

    return sum1 / (2.0 * sigma) + sum2


def proximal(sigma, zstar, z, betastar=1, beta=1, noise_in=0, noise_out=0):
     
    init = zstar + np.random.normal(0, 1, len(zstar)) * 0.01

     
    Lout = np.size(softmax(zstar, betastar))

     
    additive_noise_out = np.sqrt(noise_out) * np.random.randn(Lout)

     
    additive_noise_in = np.sqrt(noise_in / 2.0) * np.random.randn(np.size(zstar))

     
    def obj(x):
        return float(
            proximal_argument(
                sigma, zstar, z, x,
                additive_noise_in, additive_noise_out,
                betastar=float(betastar), beta=float(beta)
            )
        )

    res = optimize.minimize(obj, init, method="L-BFGS-B")
    return res.x, res.fun



def ERM_training_loss(overlaps, alpha, param_prior, param_output, samples=1000):

    q, m, sigma, q_hat, m_hat, sigma_hat = overlaps
    Q_0, T, beta, betastar, noise_in, noise_out = param_output
    Q_0, lreg, kappa, kappa_stud, posdef = param_prior
    
    outputpart = 0
    L = int(T * (T+1) / 2)

    for i in range(0,samples):
        zstar, z = np.random.multivariate_normal([0,0],[[Q_0,m],[m,q]], L).T  
        mor = proximal(sigma, zstar, z, betastar = betastar, beta = beta, noise_in=noise_in, noise_out=noise_out)[1]
        outputpart += mor

    outputpart = outputpart / samples

    J = ERM_I_precise(np.sqrt(q_hat)/m_hat, 2 * lreg /m_hat, param_prior)

    value = (sigma_hat * q - q_hat * sigma)/2 - m_hat * m - 2 * alpha * outputpart + m_hat**2 / (2*sigma_hat) * J
    return - value / 2