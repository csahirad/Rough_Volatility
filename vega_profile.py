from RBergomi import RBergomi
from utils import bsinv
from payoffs import *
from Greeks import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def calc_S(a, eta, rho, dxi):
    np.random.seed(124)
    rB = RBergomi(xis=[(0.0, 0.04 + dxi), (0.25, 0.0566 + dxi), (0.5, .0693 + dxi), (0.75, 0.0800 + dxi), (1.0, 0.0894 + dxi),
                       (1.25, 0.0980 + dxi), (1.5, 0.106 + dxi), (1.75, 0.113 + dxi), (2.0, 0.120 + dxi)],
                  n=100,
                  N=3000,
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


S_0 = calc_S(-0.3, 2.0, -0.7, 0)

alphas = [-0.2, -0.4]
etas = [1.5, 2.5]
rhos = [-0.9, -0.5]
dxis = [-0.01, 0.01]
scenarios = []
for alpha in alphas:
    scenarios.append(calc_S(alpha, 2.0, -0.7, 0))
for eta in etas:
    scenarios.append(calc_S(-0.3, eta, -0.7, 0))
for rho in rhos:
    scenarios.append(calc_S(-0.3, 2.0, rho, 0))
for dxi in dxis:
    scenarios.append(calc_S(-0.3, 2.0, -.07, dxi))

N = len(scenarios)
print(N)
# strikes = np.array([90, 95, 100, 105, 110, 115])
strikes = np.linspace(90, 110, N)

B = np.zeros((N, 1))
A = np.zeros((N, 1))
CL = np.zeros((N, 1))
AC = np.zeros((N, 1))
V = np.zeros((N, len(strikes)))

for i in range(N):
    B[i] = barrier(scenarios[i]*100, K=100, B=110)
    A[i] = asian(scenarios[i]*100, K=100)
    CL[i] = cliquet(scenarios[i]*100)
    AC[i] = autocallable(scenarios[i]*100)
    V[i] = vanilla(scenarios[i]*100, strikes).reshape(N, )

c0s = vanilla(S_0*100, strikes)

vegas = np.zeros((N, 1))
for i in range(len(c0s)):
    sigma_bs = bsinv(c0s[i], 100, strikes[i], 1)
    vegas[i] = BS_vega(100, strikes[i], 1, 0, sigma_bs)

clf = Ridge(alpha=0.1, fit_intercept=False)
clf.fit(V, A)
W = clf.coef_.reshape(N, 1)
# print(W)

# weighted_vegas = np.ones((6, 1))
weighted_vegas = vegas * W
weighted_vegas = weighted_vegas.reshape(N,)
plt.bar(strikes, weighted_vegas)
plt.xlabel('Strike')
plt.ylabel('Weighted Vega')
plt.grid('True')
plt.show()
