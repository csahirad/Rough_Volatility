from RBergomi import RBergomi
from VolSurface import VolSurface
from utils import bsinv
from payoffs import *


vec_bsinv = np.vectorize(bsinv)
k = np.arange(-0.5, 0.51, 0.01)
K = np.exp(k)[np.newaxis, :]
Ts = np.arange(0.1, 2.0, 0.1)[:, np.newaxis]
implied_vols = np.zeros((Ts.shape[0], K.shape[1]))
i = 0
np.random.seed(0)
for T in Ts:
    rB = RBergomi(xis=[(0.0, 0.235**2), (0.25, 0.235**2), (0.5, 0.235**2), (0.75, 0.235**2), (2.0, 0.235**2)],
                  n=100,
                  N=30000,
                  T=T,
                  a=-0.43,
                  eta=1.9,
                  rho=-0.9)
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    Y = rB.Y(dW1)
    dB = rB.dB(dW1, dW2)
    XI = rB.XI(interp_method='pwc')
    V = rB.V(Y, XI)
    S = rB.S(V, dB)
    call_prices = vanilla(S, K)
    implied_vols[i, :] = np.squeeze(vec_bsinv(call_prices, 1., np.transpose(K), rB.T))
    i += 1

VS = VolSurface(k, Ts, implied_vols)
VS.plot_smile(T=1.0)
