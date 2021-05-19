"""
Script used to generate training data for neural network
"""

from RBergomi import RBergomi
from Theta import *
from utils import bsinv
from payoffs import *

vec_bsinv = np.vectorize(bsinv)
k = np.linspace(-0.4, 0.4, 9)
K = np.exp(k)[np.newaxis, :]
Ts = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])[:, np.newaxis]
Tmax = np.max(Ts)
np.random.seed(124)

sims = 100
i = 0
dat = np.zeros((sims, 67))
while True:
    # Randomly select a set of model parameters
    theta = Theta(xi_ub=0.16, xi_lb=0.01, eta_ub=4.0, eta_lb=0.5, rho_ub=-0.1, rho_lb=-0.95, alpha_ub=0, alpha_lb=-0.475)
    rB = RBergomi(xis=[(0.0, theta.xi0), (0.25, theta.xi0), (0.5, theta.xi0), (0.75, theta.xi0), (1.0, theta.xi0),
                       (1.25, theta.xi0), (1.5, theta.xi0), (1.75, theta.xi0), (2.0, theta.xi0)],
                  n=100,
                  N=30000,
                  T=Tmax,
                  a=theta.alpha,
                  eta=theta.eta,
                  rho=theta.rho)
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    Y = rB.Y(dW1)
    dB = rB.dB(dW1, dW2)
    XI = rB.XI(interp_method='log-linear')
    V = rB.V(Y, XI)
    S = rB.S(V, dB)
    implied_vols = np.zeros((Ts.shape[0], K.shape[1]))
    for idx, t in enumerate(Ts):
        call_prices = vanilla(S[:, :(int(t*rB.n) + 1)], K)
        implied_vols[idx, :] = np.squeeze(vec_bsinv(call_prices, 1., np.transpose(K), t))

    # VS = VolSurface(k, Ts, implied_vols)
    # VS.plot_surface()

    if np.min(implied_vols) > 0.01:
        print(i)
        vols = np.ndarray.flatten(implied_vols)
        theta = np.array([theta.xi0, theta.alpha, theta.eta, theta.rho])
        dat[i, :] = np.concatenate((vols, theta))
        i += 1

    if i == sims:
        # np.savetxt('training_data.csv', dat, delimiter=",")
        break
