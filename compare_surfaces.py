"""
Script used to peform sensitivity analysis of model parameters
"""

import numpy as np

from RBergomi import RBergomi
from VolSurface import VolSurface
from utils import bsinv
from payoffs import *
import matplotlib.pyplot as plt

vec_bsinv = np.vectorize(bsinv)
k = np.arange(-0.4, 0.41, 0.01)
K = np.exp(k)[np.newaxis, :]
Ts = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])[:, np.newaxis]
Tmax = np.max(Ts)

np.random.seed(124)
rB_1 = RBergomi(xis=[(0.0, 0.04), (0.25, 0.0566), (0.5, .0693), (0.75, 0.0800), (1.0, 0.0894),
                     (1.25, 0.0980), (1.5, 0.106), (1.75, 0.113), (2.0, 0.120)],
                n=100,
                N=30000,
                T=Tmax,
                a=-0.3,
                eta=2.0,
                rho=-0.9)
dW1_1 = rB_1.dW1()
dW2_1 = rB_1.dW2()
Y_1 = rB_1.Y(dW1_1)
dB_1 = rB_1.dB(dW1_1, dW2_1)
XI_1 = rB_1.XI(interp_method='log-linear')
V_1 = rB_1.V(Y_1, XI_1)
S_1 = rB_1.S(V_1, dB_1)
call_prices_1 = vanilla(S_1, K)
implied_vols_1 = vec_bsinv(call_prices_1, 1, np.transpose(K), 1)
# implied_vols_1 = np.zeros((Ts.shape[0], K.shape[1]))
# for idx, t in enumerate(Ts):
#     call_prices_1 = vanilla(S_1[:, :(int(t*rB_1.n) + 1)], K)
#     implied_vols_1[idx, :] = np.squeeze(vec_bsinv(call_prices_1, 1., np.transpose(K), t))

np.random.seed(124)
rB_2 = RBergomi(xis=[(0.0, 0.04), (0.25, 0.0566), (0.5, .0693), (0.75, 0.0800), (1.0, 0.0894),
                     (1.25, 0.0980), (1.5, 0.106), (1.75, 0.113), (2.0, 0.120)],
                n=100,
                N=30000,
                T=Tmax,
                a=-0.3,
                eta=2.0,
                rho=-0.5)
dW1_2 = rB_2.dW1()
dW2_2 = rB_2.dW2()
Y_2 = rB_2.Y(dW1_2)
dB_2 = rB_2.dB(dW1_2, dW2_2)
XI_2 = rB_2.XI(interp_method='log-linear')
V_2 = rB_2.V(Y_2, XI_2)
S_2 = rB_2.S(V_2, dB_2)
call_prices_2 = vanilla(S_2, K)
implied_vols_2 = vec_bsinv(call_prices_2, 1, np.transpose(K), 1)
# implied_vols_2 = np.zeros((Ts.shape[0], K.shape[1]))
# for idx, t in enumerate(Ts):
#     call_prices_2 = vanilla(S_2[:, :(int(t*rB_2.n) + 1)], K)
#     implied_vols_2[idx, :] = np.squeeze(vec_bsinv(call_prices_2, 1., np.transpose(K), t))

# Plot the smiles on top of each other
plot, axes = plt.subplots()
axes.plot(k, implied_vols_1, 'r', lw=2)
axes.plot(k, implied_vols_2, 'b', lw=2)
axes.set_xlabel(r'$k$', fontsize=16)
axes.set_ylabel(r'$\sigma_{BS}$', fontsize=16)
axes.set_title(r'Effect of $\xi_0$ on Black Sholes Implied Volatility Smiles')
axes.legend(['Nominal Forward Variance Curve', 'Forward Variance Curve + 0.01'])
plt.grid(True)
plt.show()

# # Plot the ATM explosion
# skew_1 = np.abs((implied_vols_1[:, 1] - implied_vols_1[:, 0]) / (0.01 - -0.01))
# skew_2 = np.abs((implied_vols_2[:, 1] - implied_vols_2[:, 0]) / (0.01 - -0.01))
# plot, axes = plt.subplots()
# axes.plot(Ts, skew_1, 'r', lw=2)
# axes.plot(Ts, skew_2, 'b', lw=2)
# axes.set_xlabel(r'$T$', fontsize=16)
# axes.set_ylabel(r'$\sigma_{BS}$', fontsize=16)
# axes.set_title(r'Effect of $\alpha$ on ATM Black Sholes Implied Volatility Skew')
# axes.legend(['alpha=-0.4', 'alpha=-0.2'])
# plt.grid(True)
# plt.show()

# # Plot surfaces on top of each other
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(k, Ts)
# surf1 = ax.plot_surface(X, Y, implied_vols_1)
# surf2 = ax.plot_surface(X, Y, implied_vols_2)
# plt.show()
