"""
Script used to compute vega profiles of exotics
"""

from RBergomi import RBergomi
from utils import bsinv
from payoffs import *
from Greeks import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def calc_S(a, eta, rho, dxi):
    np.random.seed(124)
    rB = RBergomi(
        xis=[(0.0, 0.04 + dxi), (0.25, 0.0566 + dxi), (0.5, .0693 + dxi), (0.75, 0.0800 + dxi), (1.0, 0.0894 + dxi),
             (1.25, 0.0980 + dxi), (1.5, 0.106 + dxi), (1.75, 0.113 + dxi), (2.0, 0.120 + dxi)],
        n=100,
        N=10000,
        T=1,
        a=a,
        eta=eta,
        rho=rho)
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    Y = rB.Y(dW1)
    dB = rB.dB(dW1, dW2)
    XI = rB.XI(interp_method='log-linear')
    V = rB.V(Y, XI)
    S = rB.S(V, dB)
    return S


T = 1
scenarios = []

alphas = [-0.2, -0.4]
etas = [1.5, 2.5]
rhos = [-0.9, -0.5]
dxis = [-0.01, 0.01]
# alphas = np.arange(-0.4, -0.1, 0.05)
# etas = np.arange(0.5, 4.1, 0.5)
# rhos = np.arange(-0.9, 0, 0.1)
# dxis = np.arange(-0.01, 0.02, 0.01)
for alpha in alphas:
    scenarios.append(calc_S(alpha, 2.0, -0.7, 0))
for eta in etas:
    scenarios.append(calc_S(-0.3, eta, -0.7, 0))
for rho in rhos:
    scenarios.append(calc_S(-0.3, 2.0, rho, 0))
for dxi in dxis:
    scenarios.append(calc_S(-0.3, 2.0, -0.7, dxi))

N = len(scenarios)
strikes = np.linspace(70, 130, 40)
n = len(strikes)

B = np.zeros((N, 1))
A = np.zeros((N, 1))
CL = np.zeros((N, 1))
AC = np.zeros((N, 1))
V = np.zeros((N, n))
C = np.zeros((N, 1))

for i in range(N):
    B[i] = barrier(scenarios[i] * 100, K=100, B=110)
    A[i] = asian(scenarios[i] * 100, K=100)
    CL[i] = cliquet(scenarios[i] * 100, T=T)
    AC[i] = autocallable(scenarios[i] * 100, T=T)
    V[i] = vanilla(scenarios[i] * 100, strikes).reshape(n, )
    C[i] = vanilla(scenarios[i] * 100, 100)

# Compute price of hedge instruments in scenario with no perturbations
S_0 = calc_S(-0.3, 2.0, -0.7, 0)
c0s = vanilla(S_0 * 100, strikes)

# Compute vegas of heding instruments
vegas = np.zeros((n, 1))
sigma_bs = np.zeros((n, 1))
for i in range(len(c0s)):
    sigma_bs[i] = bsinv(c0s[i], 100, strikes[i], T)
    vegas[i] = BS_vega(100, strikes[i], T, 0, sigma_bs[i])

# # Compute IV of hedge instruments in each scenario
# sigs = np.zeros((N, N))
# vec_bsinv = np.vectorize(bsinv)
# for i in range(N):
#     sigs[i, :] = vec_bsinv(V[i, :], 100, strikes, 1)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.plot(range(1, 9), V)
# ax1.legend(strikes, title='Strikes')
# ax1.set(xlabel='Scenario', ylabel='Vanilla Option Price')
# ax1.set_title('Price of Hedging Instruments for Varying Scenarios')
#
# ax2.plot(strikes, V.T)
# ax2.set(xlabel='Strikes', ylabel='Vanilla Option Price')
# ax2.legend(['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5', 'Scenario 6', 'Scenario 7', 'Scenario 8'])
# ax2.set_title('Price of Hedging Instruments for Varying Scenarios')
#
# ax3.plot(strikes, sigs.T)
# ax3.set(xlabel='Strikes', ylabel='Implied Volatility')
# ax3.legend(['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5', 'Scenario 6', 'Scenario 7', 'Scenario 8'])
# ax3.set_title('IV of Hedging Instruments for Varying Scenarios')

# plt.show()

# Perform ridge regression
clf = Ridge(alpha=0.2, fit_intercept=False)
clf.fit(V, CL)
W = clf.coef_.reshape(n, 1)

# Plot vega profile weighted by hedge instruments
weighted_vegas = vegas * W
weighted_vegas = weighted_vegas.reshape(n, )
plt.bar(strikes, weighted_vegas)
# plt.xlim([25, 175])
plt.xlabel('$K$', fontsize=16)
plt.ylabel('Weighted Vega', fontsize=16)
# plt.title('Autocallable (coupon=5%, barrier=70%, \nT=1yr, freq=quarterly)', fontsize=16)
# plt.title('Asian Call (K=100, T=1yr)', fontsize=16)
plt.title('Cliquet (cap=5%, T=1yr, freq=monthly)', fontsize=16)
# plt.title('Up-and-Out Barrier Call (K=100, H=110, T=1yr)', fontsize=16)
plt.grid('True')
plt.show()
