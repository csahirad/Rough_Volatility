from RBergomi import RBergomi
from VolSurface import VolSurface
from Theta import *
from utils import bsinv
from payoffs import *
import matplotlib.pyplot as plt

vec_bsinv = np.vectorize(bsinv)
k = np.arange(-0.4, 0.41, 0.01)
K = np.exp(k)[np.newaxis, :]
Ts = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])[:, np.newaxis]
Tmax = np.max(Ts)
np.random.seed(124)
rB = RBergomi(xis=[(0.0, 0.04), (0.25, 0.0566), (0.5, .0693), (0.75, 0.0800), (1.0, 0.0894),
                   (1.25, 0.0980), (1.5, 0.106), (1.75, 0.113), (2.0, 0.120)],
              n=100,
              N=30000,
              T=Tmax,
              a=-0.3,
              eta=2.0,
              rho=-0.7)
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
VS = VolSurface(k, Ts, implied_vols)


# Plot the forward variance curve
plt.figure(1)
plt.plot(np.arange(0, Tmax, 0.1), XI(np.arange(0, Tmax, 0.1)), lw=2)
plt.xlabel('$T$ [yrs]', fontsize=16)
plt.ylabel(r'$\xi_0$', fontsize=16)
plt.grid('True')
plt.title('Initial Forward Variance Curve', fontsize=16)

# Plot some sample paths
plt.figure(2)
plt.plot(np.linspace(0, 1, 201), S.T[:, :100])
plt.xticks(np.arange(0, 1.2, step=0.2))
plt.xlabel('$T$ [yrs]', fontsize=16)
plt.ylabel('Price', fontsize=16)
plt.grid('True')
plt.title('Sample Paths of the rBergomi Price Process', fontsize=16)
plt.show()

# Plot the vol surface
VS.plot_surface()

