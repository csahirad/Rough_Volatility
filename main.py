from RBergomi import RBergomi
from VolSurface import VolSurface
from Theta import *
from utils import bsinv
from payoffs import *
import matplotlib.pyplot as plt

vec_bsinv = np.vectorize(bsinv)
k = np.arange(-0.5, 0.51, 0.01)
K = np.exp(k)[np.newaxis, :]
Ts = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])[:, np.newaxis]
Tmax = np.max(Ts)
np.random.seed(0)
theta = Theta(xi_ub=0.16, xi_lb=0.01, eta_ub=4.0, eta_lb=0.5, rho_ub=-0.1, rho_lb=-0.95, alpha_ub=0, alpha_lb=-0.475)
rB = RBergomi(xis=[(0.0, theta.xi0), (0.25, theta.xi1), (0.5, theta.xi2), (0.75, theta.xi3), (1.0, theta.xi4),
                   (1.25, theta.xi5), (1.5, theta.xi6), (1.75, theta.xi7), (2.0, theta.xi8)],
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
XI = rB.XI(interp_method='pwc')
V = rB.V(Y, XI)
S = rB.S(V, dB)
implied_vols = np.zeros((Ts.shape[0], K.shape[1]))

for idx, t in enumerate(Ts):
    call_prices = vanilla(S[:, :int(t*rB.n) + 1], K)
    implied_vols[idx, :] = np.squeeze(vec_bsinv(call_prices, 1., np.transpose(K), t))

VS = VolSurface(k, Ts, implied_vols)
VS.plot_surface()

